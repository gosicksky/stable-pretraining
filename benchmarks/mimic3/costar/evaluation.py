"""Scientific evaluation functions used for the five-seed MIMIC-III results.

These functions preserve the recorded reference experiment's model and metric
behavior. The command-line workflow lives in validate.py.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .common import (
    evaluation,
    loader,
    move,
    require,
    sha,
    state_digest,
    stream_digests,
    tensor_state,
)
from .components import OutcomeModel, batch_loss_proof, loss_statistics
from .components import base_model as make_base_model
from .data import load_mimic_counterfactual_base, iter_mimic_counterfactual_chunks

INPUT_KEYS = ("dynamic_history", "static_features", "valid_mask")
HORIZONS = 11
REPORT_HORIZONS = 6

CONFIG = {
    "covariate_dim": 25,
    "treatment_dim": 3,
    "outcome_dim": 2,
    "static_dim": 44,
    "d_model": 36,
    "num_heads": 3,
    "num_layers": 1,
    "max_relative_distance": 15,
    "ffn_hidden_dim": 144,
    "predictor_hidden_dim": 72,
    "dropout": 0.1,
    "temperature": 1.0,
    "momentum": 0.99,
    "lr": 0.001,
    "weight_decay": 0.0,
    "scale_std": 0.5,
    "shift_std": 0.5,
    "jitter_std": 0.5,
    "augmentation_probability": 0.5,
}


def metrics(sums: torch.Tensor, counts: torch.Tensor) -> dict[str, Any]:
    """Compute horizon RMSE and the diagnostic inverse-weighted MSE.

    Args:
        sums: Squared-error totals with shape [11, 2], one column per outcome.
        counts: Valid prediction-origin counts for each of the 11 horizons.

    Returns:
        Metric values and error totals, including mean RMSE over horizons 1–6.
        Outcome errors are summed without dividing by the number of outcomes.
    """
    require(
        sums.shape == (HORIZONS, 2) and counts.shape == (HORIZONS,),
        "Metric schema changed.",
    )
    require(
        bool(torch.isfinite(sums).all())
        and bool((sums >= 0).all())
        and bool((counts > 0).all()),
        "Invalid error totals.",
    )
    mse = sums.double().sum(-1) / counts.double()
    weights = torch.arange(1, HORIZONS + 1, dtype=torch.float64).reciprocal()
    weights /= weights.sum()
    rmse = mse.sqrt()
    return {
        "selection_loss": float((mse * weights).sum()),
        "mean_rmse6": float(rmse[:REPORT_HORIZONS].mean()),
        "rmse_per_horizon": rmse.tolist(),
        "mse_per_horizon": mse.tolist(),
        "squared_error_sum_per_horizon_and_outcome": sums.tolist(),
        "target_count_per_horizon": counts.tolist(),
    }


def base_model() -> Any:
    """Construct the SSL model using the recorded benchmark configuration.

    Returns:
        A newly initialized GitHub-profile model with encoder and pooling.
    """
    return make_base_model(CONFIG)


def outcome_model(
    encoder_state: dict[str, torch.Tensor],
    head_state: dict[str, torch.Tensor] | None = None,
) -> OutcomeModel:
    """Build an outcome model from an encoder and optional prediction head.

    Args:
        encoder_state: Complete encoder state dictionary, loaded strictly.
        head_state: Optional complete head state; otherwise initialize a head.

    Returns:
        An outcome model with the benchmark pooling and additive decoder.
    """
    base = base_model()
    base.encoder.load_state_dict(encoder_state, strict=True)
    model = OutcomeModel(
        base.encoder,
        base.pooling,
        d_model=36,
        treatment_dim=3,
        outcome_dim=2,
        max_horizon=HORIZONS,
        hidden_dim=128,
    )
    if head_state is not None:
        model.head.load_state_dict(head_state, strict=True)
    return model


def ssl_validation(
    model: nn.Module, dataset: Any, seed: int, device: str
) -> dict[str, Any]:
    """Evaluate SSL loss with the reference EMA and augmentation behavior.

    Args:
        model: SSL model already on the requested device. Validation updates
            its teacher EMA and advances augmentation RNG streams.
        dataset: Source validation dataset yielding SSL history batches.
        seed: Published run seed used to initialize the validation loader.
        device: Device receiving each validation batch.

    Returns:
        Patient-weighted loss, batch records, and EMA/RNG audit information.
        Online weights and module training flags are preserved.
    """
    before = state_digest(tensor_state(model.encoder))
    predictor_before = state_digest(tensor_state(model.predictor))
    flags = {name: child.training for name, child in model.named_modules()}
    updates = int(model.ema_updates)
    streams_before = stream_digests()
    total, patients, prefixes, batches = 0.0, 0, 0, []
    try:
        model.eval()
        with torch.no_grad():
            for batch in loader(dataset, 128, seed=6000 + seed):
                batch = move(batch, device)
                result = model(batch, stage="fit")
                count, valid = len(batch["valid_mask"]), int(batch["valid_mask"].sum())
                value = float(result["loss"])
                require(math.isfinite(value), "Nonfinite SSL validation loss.")
                total += value * count
                patients += count
                prefixes += valid
                batches.append({"loss": value, "patients": count, "prefixes": valid})
    finally:
        for name, child in model.named_modules():
            child.training = flags[name]
    require(
        state_digest(tensor_state(model.encoder)) == before,
        "Validation changed online encoder.",
    )
    require(
        state_digest(tensor_state(model.predictor)) == predictor_before,
        "Validation changed predictor.",
    )
    require(
        int(model.ema_updates) - updates == len(batches),
        "Validation EMA count differs.",
    )
    streams_after = stream_digests()
    require(
        streams_before["numpy"] != streams_after["numpy"],
        "Validation NumPy gates did not advance.",
    )
    return {
        "loss": total / patients,
        "patients": patients,
        "prefixes": prefixes,
        "batches": batches,
        "online_unchanged": True,
        "predictor_unchanged": True,
        "model_unchanged": False,
        "ema_updates_added": len(batches),
        "rng_restored": False,
        "rng_before": streams_before,
        "rng_after": streams_after,
    }


def factual_validation(
    model: OutcomeModel, dataset: Any, device: str
) -> dict[str, Any]:
    """Evaluate factual outcomes using the recorded checkpoint selector.

    Args:
        model: Outcome model already on the requested device.
        dataset: Factual dataset with standardized targets and target masks.
        device: Device receiving each validation batch.

    Returns:
        RMSE, SSE/counts, batch proofs, and the patient-weighted float32 batch
        loss as selection_loss. The horizon diagnostic is recorded separately.
    """
    sums = torch.zeros(HORIZONS, 2, dtype=torch.float64)
    counts = torch.zeros(HORIZONS, dtype=torch.long)
    batches, total, patients = [], 0.0, 0
    with evaluation(model):
        for batch in loader(dataset, 128, seed=7000):
            batch = move(batch, device)
            prediction = model(batch)
            proof = batch_loss_proof(prediction, batch["target"], batch["target_mask"])
            errors, available = loss_statistics(
                prediction, batch["target"], batch["target_mask"]
            )
            sums += errors.cpu()
            counts += available.cpu()
            batches.append(proof)
            count = len(batch["valid_mask"])
            total += proof["loss"] * count
            patients += count
    result = metrics(sums, counts)
    result["diagnostic_inverse_loss"] = result["selection_loss"]
    result.update(
        selection_loss=total / patients,
        release_validation_mse=total / patients,
        patients=patients,
        batches=batches,
    )
    return result


def restore(case: dict[str, Any], device: str) -> OutcomeModel:
    """Restore a complete selected model and verify its recorded hashes.

    Args:
        case: Selected-case record containing checkpoint, checkpoint_sha256,
            and selected_model_sha256.
        device: Device on which to place the restored model.

    Returns:
        An evaluation-mode outcome model with its full encoder and head state.
    """
    checkpoint = torch.load(case["checkpoint"], map_location="cpu", weights_only=True)
    require(
        sha(Path(case["checkpoint"])) == case["checkpoint_sha256"],
        "Selected checkpoint changed.",
    )
    base = base_model()
    model = OutcomeModel(
        base.encoder,
        base.pooling,
        d_model=36,
        treatment_dim=3,
        outcome_dim=2,
        max_horizon=HORIZONS,
        hidden_dim=128,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    require(
        state_digest(tensor_state(model)) == case["selected_model_sha256"],
        "Selected tensor state changed.",
    )
    return model.to(device).eval()


def counterfactual(model: OutcomeModel, device: str, seed: int) -> dict[str, Any]:
    """Evaluate counterfactual outcomes with separate mode-specific caches.

    Args:
        model: Selected outcome model already on the requested device.
        device: Device used for history encoding and outcome prediction.
        seed: Published cohort seed resolved through the configured catalog.

    Returns:
        Test RMSE and SSE/counts, using one-step cases for horizon 1 and
        sequential cases for horizons 2–11.
    """
    sums = torch.zeros(HORIZONS, 2, dtype=torch.float64)
    counts = torch.zeros(HORIZONS, dtype=torch.long)
    with evaluation(model):
        for mode in ("one_step", "sequential"):
            base = load_mimic_counterfactual_base(seed=seed, mode=mode)
            histories, masks = [], []
            for start in range(0, len(base["dynamic_history"]), 128):
                batch = move(
                    {
                        key: value[start : start + 128]
                        for key, value in base.items()
                        if key in INPUT_KEYS
                    },
                    device,
                )
                histories.append(model.encode_history(batch).cpu())
                masks.append(batch["valid_mask"].cpu())
            cached, valid = torch.cat(histories), torch.cat(masks)
            for chunk in iter_mimic_counterfactual_chunks(
                seed=seed, mode=mode, chunk_size=1024
            ):
                patients, origins = (
                    chunk["patient_index"].long(),
                    chunk["origin_index"].long(),
                )
                require(
                    bool(valid[patients, origins].all()),
                    "Invalid counterfactual origin.",
                )
                history = cached[patients, origins].unsqueeze(1).to(device)
                plans = chunk["future_treatments"].unsqueeze(1).to(device)
                target = chunk["target_standardized"].unsqueeze(1).to(device)
                mask = chunk["target_mask"].unsqueeze(1).to(device)
                prediction = model.predict_from_history(history, plans, mask)
                errors, available = loss_statistics(prediction, target, mask)
                chosen = slice(0, 1) if mode == "one_step" else slice(1, None)
                sums[chosen] += errors.cpu()[chosen]
                counts[chosen] += available.cpu()[chosen]
    return metrics(sums, counts)
