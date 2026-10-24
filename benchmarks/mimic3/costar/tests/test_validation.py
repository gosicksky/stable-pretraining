"""Synthetic checks for checkpoint selection, test gating, and saved metrics."""

import copy
import math
from pathlib import Path
from statistics import stdev

import pytest

from benchmarks.mimic3.costar import validate
from benchmarks.mimic3.costar.common import read_json, sha, write_json


pytestmark = pytest.mark.unit


def saved_records():
    records = []
    for index, seed in enumerate(validate.SEEDS, 1):
        for arm in validate.ARMS:
            for stage in validate.STAGES:
                factor = 1 if arm == "random" else (0.9 if stage == "source" else 1.1)
                rmse = index * factor
                counts = list(range(1, 12))
                records.append(
                    {
                        "seed": seed,
                        "budget": "full",
                        "arm": arm,
                        "stage": stage,
                        "metrics": {
                            "squared_error_sum_per_horizon_and_outcome": [
                                [rmse**2 * count * 0.25, rmse**2 * count * 0.75]
                                for count in counts
                            ],
                            "target_count_per_horizon": counts,
                            "mean_rmse6": rmse,
                        },
                    }
                )
    return records


def test_summary_uses_vector_rmse_equal_seed_weights_and_sample_sd(tmp_path):
    path = tmp_path / "evaluation.json"
    write_json(path, saved_records())
    summary = validate.summarize(path)
    zero, fine = summary["table"]
    assert zero["without_ssl_mean"] == pytest.approx(3)
    assert zero["without_ssl_sample_sd"] == pytest.approx(stdev([1, 2, 3, 4, 5]))
    assert zero["ssl_mean"] == pytest.approx(2.7)
    assert zero["relative_change_percent"] == pytest.approx(-10)
    assert fine["ssl_mean"] == pytest.approx(3.3)
    assert fine["relative_change_percent"] == pytest.approx(10)
    assert summary["inference_performed"] is False


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "zero_count", "nan", "aggregate"]
)
def test_summary_rejects_incomplete_or_inconsistent_saved_statistics(tmp_path, change):
    records = saved_records()
    if change == "missing":
        records.pop()
    elif change == "duplicate":
        records[-1] = copy.deepcopy(records[0])
    elif change == "zero_count":
        records[0]["metrics"]["target_count_per_horizon"][0] = 0
    elif change == "nan":
        # Infinity is representable as a JSON exponent and must still be rejected.
        records[0]["metrics"]["squared_error_sum_per_horizon_and_outcome"][0][0] = 1e300
    else:
        records[0]["metrics"]["mean_rmse6"] += 1
    path = tmp_path / "evaluation.json"
    write_json(path, records)
    if change == "nan":
        path.write_text(path.read_text().replace("1e+300", "1e999"))
    with pytest.raises(RuntimeError):
        validate.summarize(path)


def stage_fixture(directory: Path, ssl: bool):
    directory.mkdir()
    curve = []
    for epoch in range(1, 22):
        validation = {
            "loss" if ssl else "release_validation_mse": 1.0 if epoch == 1 else 2.0
        }
        curve.append(
            {"epoch": epoch, "validation": validation, "bad_epochs": epoch - 1}
        )
    write_json(directory / "curve.json", curve)
    checkpoint = directory / ("last.pt" if ssl else "best.pt")
    checkpoint.write_bytes(b"synthetic checkpoint hash fixture")
    return {
        "completed_epochs": 21,
        "bad_epochs": 20,
        "best_loss": 1.0,
        "stopping_reason": "patience20",
        "selected_epoch": 21 if ssl else 1,
        "selection": "LAST" if ssl else "complete_model_BEST",
        "validation": curve[0]["validation"],
        "curve_sha256": sha(directory / "curve.json"),
        "checkpoint_sha256": sha(checkpoint),
    }


@pytest.mark.parametrize("ssl", [False, True])
def test_stage_selection_and_first_patience_stop(tmp_path, ssl):
    directory = tmp_path / "stage"
    report = stage_fixture(directory, ssl)
    validate.check_stage(directory, report, ssl=ssl)
    report["selected_epoch"] = 20
    with pytest.raises(RuntimeError, match="selection"):
        validate.check_stage(directory, report, ssl=ssl)


def test_epoch_budget_is_checked_before_checkpoint_loading(tmp_path):
    directory = tmp_path / "stage"
    report = stage_fixture(directory, True)
    curve = read_json(directory / "curve.json") * 5
    write_json(directory / "curve.json", curve)
    report["curve_sha256"] = sha(directory / "curve.json")
    with pytest.raises(RuntimeError, match="100-epoch"):
        validate.check_stage(directory, report, ssl=True)


def mock_evaluation(monkeypatch, *, fail_validation=None, change_audit=False):
    events = []
    expected = {
        "target_count_per_horizon": [1] * 11,
        "squared_error_sum_per_horizon_and_outcome": [[1, 1]] * 11,
        "release_validation_mse": 2.0,
        "batches": [{"loss": 2.0}],
    }
    cases = [
        {
            **{key: row[key] for key in validate.IDENTITY},
            "validation": expected,
            "checkpoint_sha256": "synthetic",
        }
        for row in saved_records()
    ]
    audit_calls = []

    def audit(run):
        audit_calls.append(run)
        return {
            "status": "passed",
            "input_sha256": {
                "synthetic": "changed"
                if change_audit and len(audit_calls) > 1
                else "same"
            },
        }

    class Bundle:
        def factual(self, split):
            return split

    def factual(model, dataset, device):
        events.append("validation")
        if len(events) == fail_validation:
            return {**expected, "release_validation_mse": math.pi}
        return expected

    def counterfactual(model, device, seed):
        assert events[:20] == ["validation"] * 20
        events.append("test")
        return {"saved": True}

    monkeypatch.setattr(validate, "audit_inputs", audit)
    monkeypatch.setattr(validate, "selected_cases", lambda run: cases)
    monkeypatch.setattr(validate.data, "load_data", lambda seed: Bundle())
    monkeypatch.setattr(validate, "restore", lambda case, device: object())
    monkeypatch.setattr(validate, "factual_validation", factual)
    monkeypatch.setattr(validate, "counterfactual", counterfactual)
    monkeypatch.setattr(validate, "guard_case_inputs", lambda audit, seed: None)
    return events


def test_all_twenty_validations_precede_test_and_repeated_tests_are_refused(
    tmp_path, monkeypatch
):
    events = mock_evaluation(monkeypatch)
    validate.evaluate(tmp_path, "cpu")
    assert events == ["validation"] * 20 + ["test"] * 20
    assert read_json(tmp_path / "evaluation/complete.json")["tests"] == 20
    with pytest.raises(RuntimeError, match="already exists"):
        validate.evaluate(tmp_path, "cpu")
    assert len(events) == 40


def test_failed_validation_prevents_every_counterfactual_test(tmp_path, monkeypatch):
    events = mock_evaluation(monkeypatch, fail_validation=17)
    with pytest.raises(RuntimeError, match="selection loss"):
        validate.evaluate(tmp_path, "cpu")
    assert events == ["validation"] * 17
    assert not (tmp_path / "evaluation/test_gate.json").exists()


def test_input_change_before_test_gate_is_rejected(tmp_path, monkeypatch):
    events = mock_evaluation(monkeypatch, change_audit=True)
    with pytest.raises(RuntimeError, match="Inputs changed"):
        validate.evaluate(tmp_path, "cpu")
    assert events == ["validation"] * 20
    assert not (tmp_path / "evaluation/test_gate.json").exists()


def test_catalog_change_after_first_test_prevents_remaining_tests(
    tmp_path, monkeypatch
):
    events = mock_evaluation(monkeypatch)
    calls = []

    def guard(audit, seed):
        calls.append(seed)
        if len(calls) == 3:
            raise RuntimeError("Data catalog changed during testing.")

    monkeypatch.setattr(validate, "guard_case_inputs", guard)
    with pytest.raises(RuntimeError, match="catalog changed"):
        validate.evaluate(tmp_path, "cpu")
    assert events == ["validation"] * 20 + ["test"]
    assert not (tmp_path / "evaluation/complete.json").exists()
