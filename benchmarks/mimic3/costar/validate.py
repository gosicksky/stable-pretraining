"""Check saved COSTAR runs, replay validation, and summarize fixed tests."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch

from . import data
from .common import numerical_environment, read_json, require, sha, state_digest
from .common import utc_now, write_json
from .evaluation import counterfactual, factual_validation, restore

SEEDS = (17, 43, 44, 91, 95)
ARMS = ("random", "ssl")
STAGES = ("source", "target")
IDENTITY = ("seed", "budget", "arm", "stage")


def case_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Identify one seed, budget, initialization arm, and evaluation stage.

    Args:
        row: Saved evaluation or selected-model record.

    Returns:
        The four-part case identity.
    """
    return tuple(row[name] for name in IDENTITY)


def check_cases(rows: list[dict[str, Any]]) -> None:
    """Require each of the twenty paired evaluation cases exactly once.

    Args:
        rows: Saved evaluation or selected-model records.
    """
    expected = {
        (seed, "full", arm, stage) for seed in SEEDS for arm in ARMS for stage in STAGES
    }
    require(len(rows) == len(expected), "Expected exactly 20 cases.")
    require({case_key(row) for row in rows} == expected, "Missing or duplicate cases.")


def selected_cases(run: Path) -> list[dict[str, Any]]:
    """Read selections and resolve checkpoints inside the supplied run.

    Args:
        run: Directory containing certified development and stage reports.

    Returns:
        Twenty selected-model records with relocated checkpoint paths.
    """
    rows = []
    for record in read_json(run / "development.json"):
        seed = record["seed"]
        require(seed in SEEDS, "Unsupported seed.")
        for arm in ARMS:
            for stage in STAGES:
                saved = record["budgets"]["full"][arm][stage]
                directory = run / f"seed{seed}" / "full" / arm / stage
                report = read_json(directory / "report.json")
                require(
                    saved == report, "Development selection and stage report differ."
                )
                require(
                    report["seed"] == seed and report["stage"] == stage,
                    "Stage identity changed.",
                )
                rows.append(
                    {
                        **report,
                        "budget": "full",
                        "arm": arm,
                        "checkpoint": str(directory / "best.pt"),
                    }
                )
    check_cases(rows)
    return rows


def check_stage(
    directory: Path, report: dict[str, Any], *, ssl: bool
) -> dict[str, str]:
    """Check first early stopping and validation-selected checkpoint provenance.

    Args:
        directory: Directory containing the stage curve and checkpoint.
        report: Recorded stage metadata and expected hashes.
        ssl: Whether the stage selects the last SSL epoch instead of supervised BEST.

    Returns:
        Hashes of the verified curve and selected checkpoint.
    """
    curve_path = directory / "curve.json"
    require(sha(curve_path) == report["curve_sha256"], "Training curve changed.")
    curve = read_json(curve_path)
    require(0 < len(curve) <= 100, "A stage exceeds the 100-epoch budget.")
    require(report["completed_epochs"] == len(curve), "Completed epoch count changed.")
    best, best_epoch, bad = math.inf, 0, 0
    for epoch, point in enumerate(curve, 1):
        require(point["epoch"] == epoch and bad < 20, "Invalid epoch/early-stop order.")
        loss = point["validation"]["loss" if ssl else "release_validation_mse"]
        require(math.isfinite(loss), "Nonfinite validation loss.")
        if loss < best:
            best, best_epoch, bad = loss, epoch, 0
        else:
            bad += 1
        require(point["bad_epochs"] == bad, "Early-stop counter changed.")
    require(bad == 20 or len(curve) == 100, "Training stopped before either limit.")
    if ssl:
        require(bad == 20, "Reused SSL must have reached its recorded patience20 stop.")
    require(
        report["bad_epochs"] == bad and report["best_loss"] == best,
        "Final training state changed.",
    )
    require(
        report["stopping_reason"] == ("patience20" if bad == 20 else "epoch_cap"),
        "Stopping reason changed.",
    )
    selected = len(curve) if ssl else best_epoch
    require(report["selected_epoch"] == selected, "Wrong checkpoint selection.")
    require(
        report["selection"] == ("LAST" if ssl else "complete_model_BEST"),
        "Wrong checkpoint selection policy.",
    )
    if not ssl:
        require(
            report["validation"] == curve[selected - 1]["validation"],
            "Selected validation record changed.",
        )
    checkpoint = directory / ("last.pt" if ssl else "best.pt")
    require(sha(checkpoint) == report["checkpoint_sha256"], "Checkpoint changed.")
    return {
        str(curve_path): report["curve_sha256"],
        str(checkpoint): report["checkpoint_sha256"],
    }


def audit_inputs(run: Path) -> dict[str, Any]:
    """Verify certificates, selections, data files, and complete model transfer.

    Args:
        run: Directory containing the completed five-seed training artifacts.

    Returns:
        Audit status and the hashes of all checked inputs; no inference is performed.
    """
    certificate = read_json(run / "verification.json")
    require(
        certificate["status"] == "passed", "The recorded artifact audit did not pass."
    )
    files = {}
    for name in ("development", "protocol", "evaluation_plan"):
        path = run / f"{name}.json"
        require(
            sha(path) == certificate[f"{name}_sha256"], "Recorded certificate is stale."
        )
        files[str(path)] = sha(path)
    plan = read_json(run / "evaluation_plan.json")
    development = {row["seed"]: row for row in read_json(run / "development.json")}
    cases = selected_cases(run)
    by_key = {case_key(case): case for case in cases}
    for seed in SEEDS:
        ssl_dir = run / f"seed{seed}" / "ssl"
        ssl_report = read_json(ssl_dir / "report.json")
        require(
            ssl_report == development[seed]["ssl"], "Recorded SSL selection changed."
        )
        files.update(check_stage(ssl_dir, ssl_report, ssl=True))
        source_heads = []
        for arm in ARMS:
            source = by_key[(seed, "full", arm, "source")]
            target = by_key[(seed, "full", arm, "target")]
            source_heads.append(source["initial_head_sha256"])
            for case in (source, target):
                directory = Path(case["checkpoint"]).parent
                files.update(check_stage(directory, case, ssl=False))
                checkpoint = torch.load(
                    case["checkpoint"], map_location="cpu", weights_only=True
                )
                require(
                    checkpoint["epoch"] == case["selected_epoch"],
                    "Checkpoint epoch differs.",
                )
                require(
                    state_digest(checkpoint["state_dict"])
                    == case["selected_model_sha256"],
                    "Selected model tensor state changed.",
                )
            initial_path = Path(target["checkpoint"]).parent / "initial.pt"
            initial = torch.load(initial_path, map_location="cpu", weights_only=True)
            initial_state = initial.get("state_dict", initial)
            require(
                state_digest(initial_state)
                == source["selected_model_sha256"]
                == target["initial_model_sha256"],
                "Full source-to-target transfer changed.",
            )
            files[str(initial_path)] = sha(initial_path)
        require(
            source_heads[0] == source_heads[1],
            "Initial prediction heads are not paired.",
        )
        record = data.cohort_record(seed)
        require(
            record["data_manifest_sha256"]
            == plan["data_manifests"][str(seed)]["sha256"],
            "Data manifest differs from the certified experiment.",
        )
        manifest = data.read_manifest(seed)
        root = Path(record["data_dir"])
        files[str(Path(record["data_manifest"]))] = record["data_manifest_sha256"]
        # Hash test arrays without loading their values or invoking a model.
        for group in manifest["array_groups"].values():
            for item in group.values():
                path = (root / item["path"]).resolve()
                require(
                    path.is_relative_to(root) and path.suffix == ".npy",
                    "Invalid array path.",
                )
                require(
                    path.stat().st_size == item["size_bytes"]
                    and sha(path) == item["sha256"],
                    "Prepared data changed.",
                )
                files[str(path)] = item["sha256"]
    require(data.DATA_CATALOG is not None, "Set --data-catalog first.")
    files[str(data.DATA_CATALOG)] = sha(data.DATA_CATALOG)
    for path in Path(__file__).parent.glob("*.py"):
        files[str(path.resolve())] = sha(path)
    from inspect import getfile
    from stable_pretraining.methods.costar import COSTAREncoder
    from stable_pretraining.losses.costar import COSTARLoss

    for item in (COSTAREncoder, COSTARLoss):
        path = Path(getfile(item)).resolve()
        files[str(path)] = sha(path)
    return {
        "status": "passed",
        "utc": utc_now(),
        "cases": 20,
        "inference_performed": False,
        "input_sha256": files,
    }


def compare_validation(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    """Require a factual replay to reproduce its recorded batch statistics.

    Args:
        actual: Statistics computed by the new factual validation pass.
        expected: Statistics saved when selecting the supervised checkpoint.
    """
    require(
        actual["target_count_per_horizon"] == expected["target_count_per_horizon"],
        "Validation replay counts changed.",
    )
    field = "squared_error_sum_per_horizon_and_outcome"
    require(
        torch.allclose(
            torch.tensor(actual[field], dtype=torch.float64),
            torch.tensor(expected[field], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-8,
        ),
        "Validation replay SSE changed.",
    )
    require(
        actual["release_validation_mse"] == expected["release_validation_mse"],
        "Validation float32 selection loss changed.",
    )
    require(
        actual["batches"] == expected["batches"], "Validation batch proofs changed."
    )


def guard_case_inputs(audit: dict[str, Any], seed: int) -> None:
    """Keep each test bound to the catalog and manifest verified before the gate.

    Args:
        audit: Passed input audit containing immutable file hashes.
        seed: Published simulation seed for this test case.
    """
    require(data.DATA_CATALOG is not None, "Set --data-catalog first.")
    expected = audit["input_sha256"]
    require(
        sha(data.DATA_CATALOG) == expected[str(data.DATA_CATALOG)],
        "Data catalog changed during testing.",
    )
    record = data.cohort_record(seed)
    manifest = Path(record["data_manifest"])
    require(
        sha(manifest) == expected[str(manifest)],
        "Data manifest changed during testing.",
    )


def evaluate(run: Path, device: str) -> None:
    """Replay all factual validations before running the fixed target tests once.

    Args:
        run: Completed training directory without an existing evaluation directory.
        device: PyTorch device used for model inference.
    """
    output = run / "evaluation"
    require(
        not output.exists(),
        "Evaluation already exists; use summarize, not repeated inference.",
    )
    audit = audit_inputs(run)
    selected = selected_cases(run)
    output.mkdir()  # Atomic claim; an interrupted or competing run cannot repeat tests.
    write_json(output / "input_audit.json", audit)
    bundles, replays, results, events = {}, [], [], []
    for case in selected:
        seed = case["seed"]
        if seed not in bundles:
            bundles[seed] = data.load_data(seed)
        model = restore(case, device)
        split = "source_val" if case["stage"] == "source" else "target_val"
        actual = factual_validation(model, bundles[seed].factual(split), device)
        compare_validation(actual, case["validation"])
        identity = {name: case[name] for name in IDENTITY}
        replays.append({**identity, "status": "passed", "metrics": actual})
        events.append(
            {**identity, "event": "validation_replay_passed", "utc": utc_now()}
        )
        write_json(output / "validation_replay.json", replays)
        write_json(output / "events.json", events)
        del model
    require(len(replays) == 20, "All validation replays must pass before testing.")
    # Recheck after validation so changed inputs cannot pass through a stale gate.
    refreshed = audit_inputs(run)
    require(
        refreshed["input_sha256"] == audit["input_sha256"],
        "Inputs changed during validation.",
    )
    write_json(
        output / "test_gate.json", {"status": "passed", "validation_replays": 20}
    )
    for case in selected:
        guard_case_inputs(audit, case["seed"])
        identity = {name: case[name] for name in IDENTITY}
        events.append({**identity, "event": "test_started", "utc": utc_now()})
        write_json(output / "events.json", events)
        model = restore(case, device)
        actual = counterfactual(model, device, case["seed"])
        guard_case_inputs(audit, case["seed"])
        results.append(
            {
                **identity,
                "checkpoint_sha256": case["checkpoint_sha256"],
                "metrics": actual,
            }
        )
        events.append({**identity, "event": "test_completed", "utc": utc_now()})
        write_json(output / "evaluation.json", results)
        write_json(output / "events.json", events)
        del model
    write_json(output / "complete.json", {"status": "complete", "tests": len(results)})


def summarize(evaluation_path: Path) -> dict[str, Any]:
    """Recompute the PR table from saved squared-error sums and counts.

    Args:
        evaluation_path: JSON file containing all twenty aggregate evaluation records.

    Returns:
        Full-precision per-seed values, group means, sample SDs, and relative changes.
    """
    records = read_json(evaluation_path)
    check_cases(records)
    values = {}
    seed_results = []
    for record in records:
        metric = record["metrics"]
        sse = metric["squared_error_sum_per_horizon_and_outcome"]
        counts = metric["target_count_per_horizon"]
        require(len(sse) == len(counts) == 11, "Expected eleven saved horizons.")
        rmse = []
        for pair, count in zip(sse, counts, strict=True):
            require(
                len(pair) == 2 and all(math.isfinite(x) and x >= 0 for x in pair),
                "Invalid per-outcome SSE.",
            )
            require(
                isinstance(count, int) and not isinstance(count, bool) and count > 0,
                "Invalid valid-origin count.",
            )
            rmse.append(math.sqrt(sum(pair) / count))
        value = mean(rmse[:6])
        if "mean_rmse6" in metric:
            require(
                math.isclose(value, metric["mean_rmse6"], rel_tol=1e-12, abs_tol=1e-12),
                "Saved aggregate disagrees with SSE/counts.",
            )
        values[case_key(record)] = value
        seed_results.append(
            {
                **{name: record[name] for name in IDENTITY},
                "rmse_h1_h6": value,
                "rmse_per_horizon": rmse,
            }
        )
    table = []
    for stage, label in (
        ("source", "Zero-shot"),
        ("target", "After target fine-tuning"),
    ):
        random = [values[(seed, "full", "random", stage)] for seed in SEEDS]
        ssl = [values[(seed, "full", "ssl", stage)] for seed in SEEDS]
        require(mean(random) > 0, "Relative RMSE change requires a positive baseline.")
        table.append(
            {
                "evaluation": label,
                "without_ssl_mean": mean(random),
                "without_ssl_sample_sd": stdev(random),
                "ssl_mean": mean(ssl),
                "ssl_sample_sd": stdev(ssl),
                "relative_change_percent": 100 * (mean(ssl) / mean(random) - 1),
            }
        )
    return {
        "seeds": list(SEEDS),
        "sample_sd_ddof": 1,
        "inference_performed": False,
        "table": table,
        "seed_results": seed_results,
    }


def format_table(summary: dict[str, Any]) -> str:
    """Format aggregate statistics as the four-column PR results table.

    Args:
        summary: Dictionary returned by summarize.

    Returns:
        Markdown table with the published rounding.
    """
    lines = [
        "| Target-domain evaluation | Without SSL | With SSL pretraining | Relative RMSE change |",
        "|---|---:|---:|---:|",
    ]
    for row in summary["table"]:
        lines.append(
            f"| {row['evaluation']} | {row['without_ssl_mean']:.5f} ± "
            f"{row['without_ssl_sample_sd']:.5f} | {row['ssl_mean']:.5f} ± "
            f"{row['ssl_sample_sd']:.5f} | {row['relative_change_percent']:+.3f}% |"
        )
    return "\n".join(lines)


def main() -> None:
    """Run the check, evaluate, or summarize command."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
        command.add_argument("--data-catalog", type=Path, required=True)
        if name == "evaluate":
            command.add_argument("--device", default="cuda")
    summary = commands.add_parser("summarize")
    summary.add_argument("--evaluation", type=Path, required=True)
    summary.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "summarize":
        result = summarize(args.evaluation)
        if args.output:
            require(not args.output.exists(), "Summary output already exists.")
            write_json(args.output, result)
        print(format_table(result))
        return
    data.configure_catalog(args.data_catalog)
    run = args.run_dir.resolve()
    if args.command == "check":
        audit = audit_inputs(run)
        print(f"Verified {audit['cases']} selected models; no inference performed.")
    else:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        numerical_environment()
        evaluate(run, args.device)


if __name__ == "__main__":
    main()
