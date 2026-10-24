"""Exercise portable catalog paths and integrity guards using invented data."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.mimic3.costar import data


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def reset_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the explicitly selected private catalog isolated between tests."""
    monkeypatch.setattr(data, "DATA_CATALOG", None)


def synthetic_manifest(seed: int) -> dict:
    """Create schema metadata with fictional identifiers and no clinical values."""
    pool = [f"invented-{seed}-target-{index}" for index in range(10)]
    splits = {
        split: {
            "count": 1,
            "patient_hashes": [f"invented-{seed}-{split}"],
            "valid_lengths": [3],
        }
        for split in data.FACTUAL_SPLITS
    }
    splits["target_train"]["patient_hashes"] = pool[:1]
    splits["target_train_pool"] = {"count": 10, "patient_hashes": pool}
    test_ids = [f"invented-{seed}-test"]
    return {
        "schema_version": 1,
        "seed": seed,
        "data_seed": seed,
        "horizons": 11,
        "few_shot_fraction": 0.1,
        "few_shot_selection_indices": [0],
        "dimensions": {
            "covariate_dim": 25,
            "treatment_dim": 3,
            "outcome_dim": 2,
            "static_dim": 44,
            "history_length": 99,
        },
        "alignment_profile": "official_release",
        "time_alignment": "Synthetic fixture for cached X_(j+1) metadata.",
        "split_patient_sets_disjoint": True,
        "splits": splits,
        "counterfactual": {
            "patient_count": 1,
            "patient_hashes": test_ids,
            "one_step": {"patient_hashes": test_ids},
            "sequential": {"patient_hashes": test_ids},
        },
        "normalization": {"output_means": [0, 0], "output_stds": [1, 1], "ddof": 1},
    }


@pytest.fixture
def catalog(tmp_path: Path) -> Path:
    """Write five tiny synthetic manifests with catalog-relative paths."""
    cohorts = {}
    for seed in data.PUBLISHED_SEEDS:
        folder = tmp_path / "arrays" / f"seed{seed}"
        folder.mkdir(parents=True)
        manifest = folder / "data_manifest.json"
        manifest.write_text(json.dumps(synthetic_manifest(seed)), encoding="utf-8")
        cohorts[str(seed)] = {
            "data_dir": folder.relative_to(tmp_path).as_posix(),
            "data_manifest": manifest.relative_to(tmp_path).as_posix(),
            "data_manifest_sha256": data.sha256(manifest),
        }
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "published_seeds": list(data.PUBLISHED_SEEDS),
                "cohorts": cohorts,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_requires_explicit_catalog() -> None:
    with pytest.raises(ValueError, match="configure_catalog"):
        data.read_catalog()


def test_relative_paths_are_bound_to_catalog(
    catalog: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(catalog.parent)
    data.configure_catalog(Path("catalog.json"))
    elsewhere = tmp_path / "different-cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    record = data.cohort_record(17)
    assert Path(record["data_dir"]) == catalog.parent / "arrays/seed17"
    assert (
        Path(record["data_manifest"]) == Path(record["data_dir"]) / "data_manifest.json"
    )
    assert data.read_manifest(17)["seed"] == 17


def test_manifest_hash_change_is_rejected(catalog: Path) -> None:
    data.configure_catalog(catalog)
    manifest = Path(data.cohort_record(17)["data_manifest"])
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest changed"):
        data.read_manifest(17)


def test_manifest_cannot_be_rebound_to_another_directory(catalog: Path) -> None:
    value = json.loads(catalog.read_text(encoding="utf-8"))
    value["cohorts"]["17"]["data_dir"] = "arrays/seed43"
    catalog.write_text(json.dumps(value), encoding="utf-8")
    data.configure_catalog(catalog)
    with pytest.raises(ValueError, match="directory"):
        data.cohort_record(17)


@pytest.mark.parametrize("corruption", ["overlap", "selection", "normalization"])
def test_cohort_guards_reject_invalid_metadata(corruption: str) -> None:
    manifest = copy.deepcopy(synthetic_manifest(17))
    if corruption == "overlap":
        manifest["splits"]["source_val"]["patient_hashes"] = manifest["splits"][
            "source_train"
        ]["patient_hashes"]
    elif corruption == "selection":
        manifest["few_shot_selection_indices"] = [9]
    else:
        manifest["normalization"]["output_stds"] = [0, 1]
    with pytest.raises(ValueError):
        data.validate_cohorts(manifest)


def test_array_hash_change_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "toy.npy"
    np.save(path, np.zeros((2, 3), dtype=np.float32))
    manifest = {
        "array_groups": {
            "fixture": {
                "toy": {
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": data.sha256(path),
                    "shape": [2, 3],
                    "dtype": "float32",
                }
            }
        }
    }
    np.testing.assert_array_equal(
        data.load_group(tmp_path, manifest, "fixture")["toy"], 0
    )
    np.save(path, np.ones((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="Certified numerical array changed"):
        data.load_group(tmp_path, manifest, "fixture")
