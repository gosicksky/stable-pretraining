"""Read five locally provided, certified MIMIC cohorts for result validation.

No author dataset/model implementation is imported. Each cohort keeps its own
patient split, source-fitted outcome statistics and ten-percent target subset.
Configure the private catalog explicitly; no dataset is bundled or downloaded.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


DATA_CATALOG: Path | None = None
PUBLISHED_SEEDS = (17, 43, 44, 91, 95)
FACTUAL_SPLITS = ("source_train", "source_val", "target_train", "target_val")
FACTUAL_KEYS = (
    "dynamic_history",
    "static_features",
    "valid_mask",
    "current_treatments",
    "next_outcomes",
)
HORIZONS = 11


def configure_catalog(path: Path) -> None:
    """Select the local schema-1 catalog used by the evaluation data readers.

    Args:
        path: Existing catalog JSON. Relative paths are resolved when configured.

    Raises:
        FileNotFoundError: The selected catalog does not exist or is not a file.
    """
    catalog = Path(path).expanduser().resolve()
    if not catalog.is_file():
        raise FileNotFoundError(f"MIMIC data catalog is not a file: {catalog}")
    global DATA_CATALOG
    DATA_CATALOG = catalog


def _catalog_path() -> Path:
    if DATA_CATALOG is None:
        raise ValueError(
            "No MIMIC data catalog configured. Call configure_catalog(path) "
            "with your local schema-1 catalog before reading data."
        )
    return DATA_CATALOG


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def read_catalog() -> dict[str, Any]:
    catalog = json.loads(_catalog_path().read_text(encoding="utf-8-sig"))
    if (
        catalog["schema_version"] != 1
        or catalog["published_seeds"] != list(PUBLISHED_SEEDS)
        or set(catalog["cohorts"]) != {str(seed) for seed in PUBLISHED_SEEDS}
    ):
        raise ValueError(
            "The complete five-cohort catalog must be certified before training."
        )
    return catalog


def cohort_record(seed: int) -> dict[str, Any]:
    if seed not in PUBLISHED_SEEDS or isinstance(seed, bool):
        raise ValueError("Use one published MIMIC seed: 17,43,44,91,95.")
    record = read_catalog()["cohorts"][str(seed)]
    catalog_dir = _catalog_path().parent
    data_dir, manifest = (
        (catalog_dir / record["data_dir"]).resolve(),
        (catalog_dir / record["data_manifest"]).resolve(),
    )
    if (
        manifest != data_dir / "data_manifest.json"
        or sha256(manifest) != record["data_manifest_sha256"]
    ):
        raise ValueError(
            "A catalog cohort manifest changed or disagrees with its directory."
        )
    return {**record, "data_dir": str(data_dir), "data_manifest": str(manifest)}


def read_manifest(seed: int) -> dict[str, Any]:
    record = cohort_record(seed)
    manifest = json.loads(Path(record["data_manifest"]).read_text(encoding="utf-8-sig"))
    if (
        manifest["schema_version"] != 1
        or manifest["seed"] != seed
        or manifest["data_seed"] != seed
        or manifest["horizons"] != HORIZONS
        or manifest["few_shot_fraction"] != 0.1
        or manifest["dimensions"]
        != {
            "covariate_dim": 25,
            "treatment_dim": 3,
            "outcome_dim": 2,
            "static_dim": 44,
            "history_length": 99,
        }
        or manifest.get("alignment_profile") != "official_release"
        or "cached X_(j+1)" not in manifest["time_alignment"]
        or not manifest["split_patient_sets_disjoint"]
    ):
        raise ValueError(
            "Unsupported cohort, dimensions, or official cached-next-X history semantics."
        )
    validate_cohorts(manifest)
    return manifest


def validate_cohorts(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate isolation and normalization without assuming seed17 counts."""
    seen: set[str] = set()
    counts = {}
    for name in FACTUAL_SPLITS:
        split = manifest["splits"][name]
        count, ids = split["count"], split["patient_hashes"]
        lengths = np.asarray(split["valid_lengths"])
        if (
            not isinstance(count, int)
            or count < 1
            or len(ids) != count
            or len(set(ids)) != count
            or seen.intersection(ids)
            or lengths.shape != (count,)
            or not np.issubdtype(lengths.dtype, np.integer)
            or np.any((lengths < 1) | (lengths > 99))
        ):
            raise ValueError(f"Invalid or overlapping patient cohort: {name}.")
        seen.update(ids)
        counts[name] = count
    pool = manifest["splits"]["target_train_pool"]
    pool_ids, pool_count = pool["patient_hashes"], pool["count"]
    selected = np.asarray(manifest["few_shot_selection_indices"], dtype=np.int64)
    chosen_count = int(pool_count * manifest["few_shot_fraction"])
    if (
        pool_count < 1
        or len(pool_ids) != pool_count
        or len(set(pool_ids)) != pool_count
        or selected.shape != (chosen_count,)
        or np.any((selected < 0) | (selected >= pool_count))
        or len(np.unique(selected)) != chosen_count
        or counts["target_train"] != chosen_count
    ):
        raise ValueError("Invalid per-cohort ten-percent target training selection.")
    if [pool_ids[int(i)] for i in selected] != manifest["splits"]["target_train"][
        "patient_hashes"
    ]:
        raise ValueError(
            "Selected target patient order differs from the cached cohort."
        )
    outside_target = set().union(
        *(
            manifest["splits"][name]["patient_hashes"]
            for name in ("source_train", "source_val", "target_val")
        )
    )
    if outside_target.intersection(pool_ids):
        raise ValueError("The target training pool overlaps another development split.")
    cf = manifest["counterfactual"]
    test_ids, test_count = cf["patient_hashes"], cf["patient_count"]
    if (
        not isinstance(test_count, int)
        or test_count < 1
        or len(test_ids) != test_count
        or len(set(test_ids)) != test_count
        or seen.intersection(test_ids)
        or set(pool_ids).intersection(test_ids)
        or cf["one_step"]["patient_hashes"] != test_ids
        or cf["sequential"]["patient_hashes"] != test_ids
    ):
        raise ValueError(
            "Counterfactual patients overlap or mode-specific identity order changed."
        )
    norm = manifest["normalization"]
    means, scales = np.asarray(norm["output_means"]), np.asarray(norm["output_stds"])
    if (
        means.shape != (2,)
        or scales.shape != (2,)
        or not np.isfinite(means).all()
        or not np.isfinite(scales).all()
        or not (scales > 0).all()
        or norm["ddof"] != 1
    ):
        raise ValueError("Per-cohort full-source outcome statistics are invalid.")
    return {"counts": counts, "test_count": test_count, "pairwise_disjoint": True}


def load_group(
    data_dir: Path, manifest: Mapping[str, Any], group: str
) -> dict[str, np.ndarray]:
    root, arrays = Path(data_dir).resolve(), {}
    for key, record in manifest["array_groups"][group].items():
        path = (root / record["path"]).resolve()
        if (
            not path.is_relative_to(root)
            or path.suffix != ".npy"
            or path.stat().st_size != record["size_bytes"]
            or sha256(path) != record["sha256"]
        ):
            raise ValueError(f"Certified numerical array changed: {group}/{key}.")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(value.shape) != record["shape"] or str(value.dtype) != record["dtype"]:
            raise ValueError(f"Shape or dtype changed: {group}/{key}.")
        arrays[key] = value
    return arrays


class MimicDataset(Dataset[dict[str, torch.Tensor]]):
    """History cached X_(j+1)/A_(j-1)/Y_j, planned A_(j+h) for Y_(j+h+1)."""

    def __init__(
        self,
        arrays: Mapping[str, np.ndarray],
        *,
        horizons: int = HORIZONS,
        ssl_only: bool = False,
    ) -> None:
        if set(arrays) != set(FACTUAL_KEYS) or horizons != HORIZONS:
            raise ValueError("Expected canonical factual schema and eleven horizons.")
        self.arrays, self.horizons, self.ssl_only = dict(arrays), horizons, ssl_only
        dynamic, valid, static = (
            arrays["dynamic_history"],
            arrays["valid_mask"],
            arrays["static_features"],
        )
        if (
            dynamic.ndim != 3
            or dynamic.shape[-1] != 30
            or valid.shape != dynamic.shape[:2]
            or valid.dtype != np.bool_
            or not valid[:, 0].all()
            or np.any(valid[:, 1:] & ~valid[:, :-1])
            or static.shape != (len(dynamic), 44)
            or arrays["current_treatments"].shape != (*valid.shape, 3)
            or arrays["next_outcomes"].shape != (*valid.shape, 2)
        ):
            raise ValueError("Factual history, mask, plan, or static shape mismatch.")
        for key in ("dynamic_history", "current_treatments", "next_outcomes"):
            value = arrays[key]
            if (
                value.dtype != np.float32
                or not np.isfinite(value).all()
                or np.any(value[~valid] != 0)
            ):
                raise ValueError(
                    "Factual arrays must be finite float32 with zero padding."
                )
        if static.dtype != np.float32 or not np.isfinite(static).all():
            raise ValueError("Static features must be finite float32.")
        length = valid.shape[1]
        indices = np.arange(length)[:, None] + np.arange(horizons)[None, :]
        self.future_indices, self.future_in_range = (
            np.minimum(indices, length - 1),
            indices < length,
        )

    def __len__(self) -> int:
        return self.arrays["valid_mask"].shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        result = {
            name: torch.from_numpy(self.arrays[name][index].copy())
            for name in ("dynamic_history", "static_features", "valid_mask")
        }
        if self.ssl_only:
            return result
        mask = (
            self.arrays["valid_mask"][index][self.future_indices] & self.future_in_range
        )
        future = self.arrays["current_treatments"][index][self.future_indices].copy()
        targets = self.arrays["next_outcomes"][index][self.future_indices].copy()
        future[~mask], targets[~mask] = 0, 0
        result.update(
            future_treatments=torch.from_numpy(future),
            target=torch.from_numpy(targets),
            target_mask=torch.from_numpy(mask),
        )
        return result


def build_source_label_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Bind all source patients and the cohort's independently fitted scales."""
    validate_cohorts(manifest)
    source = manifest["splits"]["source_train"]
    indices = list(range(source["count"]))
    lengths = np.asarray(source["valid_lengths"], dtype=np.int64)
    patients = source["patient_hashes"]
    counts = [int(np.maximum(lengths - h, 0).sum()) for h in range(HORIZONS)]
    return {
        "schema_version": 1,
        "data_seed": manifest["data_seed"],
        "array_groups_canonical_sha256": canonical_sha256(manifest["array_groups"]),
        "full_source_count": source["count"],
        "selection_method": "All source-training patient indices in certified cached order",
        "selections": {
            "full": {
                "count": source["count"],
                "fraction": 1.0,
                "nominal_fraction": 1.0,
                "selected_indices": indices,
                "selected_indices_sha256": canonical_sha256(indices),
                "patient_hashes": patients,
                "patient_hashes_sha256": canonical_sha256(patients),
                "valid_origins_per_horizon": counts,
                "valid_origin_horizon_pairs": sum(counts),
            }
        },
        "normalization": manifest["normalization"],
        "normalization_canonical_sha256": canonical_sha256(manifest["normalization"]),
        "fixed_cohort_hashes": {
            name: canonical_sha256(manifest["splits"][name]["patient_hashes"])
            for name in FACTUAL_SPLITS
        }
        | {
            "counterfactual": canonical_sha256(
                manifest["counterfactual"]["patient_hashes"]
            )
        },
        "exposure_scope": "Full source supervision and full source-history SSL. Target supervision uses each cohort's fixed ten-percent subset. Shared source-fitted outcome normalization is specific to the cohort. All five published populations and earlier results have been inspected; this is not a new independent holdout.",
    }


@dataclass(frozen=True)
class DataBundle:
    """One certified cohort with its own scales and full source supervision."""

    data_dir: Path
    datasets: dict[str, MimicDataset]
    normalization: dict[str, torch.Tensor]
    manifest: dict[str, Any]

    def factual(self, split: str) -> MimicDataset:
        return self.datasets[split]

    def supervised_source(self, budget: str) -> MimicDataset:
        if budget != "full":
            raise ValueError("This experiment supervises all source patients.")
        return self.datasets["source_train"]

    def ssl(self, split: str) -> MimicDataset:
        return MimicDataset(self.datasets[split].arrays, ssl_only=True)


def load_data(seed: int) -> DataBundle:
    record, manifest = cohort_record(seed), read_manifest(seed)
    root = Path(record["data_dir"]).resolve()
    datasets = {
        name: MimicDataset(load_group(root, manifest, name)) for name in FACTUAL_SPLITS
    }
    for name, dataset in datasets.items():
        if len(dataset) != manifest["splits"][name]["count"] or not np.array_equal(
            dataset.arrays["valid_mask"].sum(1),
            np.asarray(manifest["splits"][name]["valid_lengths"]),
        ):
            raise ValueError(f"Factual count or valid lengths differ: {name}.")
    normalization = {
        name: torch.tensor(manifest["normalization"][name], dtype=torch.float64)
        for name in ("output_means", "output_stds")
    }
    return DataBundle(root, datasets, normalization, manifest)


def load_mimic_counterfactual_base(*, seed: int, mode: str) -> dict[str, torch.Tensor]:
    if mode not in ("one_step", "sequential"):
        raise ValueError("Counterfactual mode must be one_step or sequential.")
    record, manifest = cohort_record(seed), read_manifest(seed)
    arrays = load_group(
        Path(record["data_dir"]), manifest, "counterfactual_base_" + mode
    )
    return {name: torch.from_numpy(value.copy()) for name, value in arrays.items()}


def iter_mimic_counterfactual_chunks(
    *, seed: int, mode: str, chunk_size: int = 1024
) -> Iterator[dict[str, torch.Tensor]]:
    if mode not in ("one_step", "sequential") or chunk_size < 1:
        raise ValueError("Require a counterfactual mode and positive chunk size.")
    record, manifest = cohort_record(seed), read_manifest(seed)
    arrays = load_group(Path(record["data_dir"]), manifest, "counterfactual_" + mode)
    for first in range(0, len(arrays["patient_index"]), chunk_size):
        chunk = {
            name: torch.from_numpy(value[first : first + chunk_size].copy())
            for name, value in arrays.items()
        }
        present = chunk["future_treatments"].shape[1]
        if present < HORIZONS:
            for name in ("future_treatments", "target_standardized", "target_mask"):
                value = chunk[name]
                padded = value.new_zeros((len(value), HORIZONS, *value.shape[2:]))
                padded[:, :present] = value
                chunk[name] = padded
        yield chunk
