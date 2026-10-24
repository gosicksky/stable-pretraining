"""Independent PyTorch components for the pinned GitHub MIMIC execution profile.

The repository encoder and loss are reused. Author implementation
files are reference text and are never imported or executed by this module.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from stable_pretraining.losses.costar import COSTARLoss
from stable_pretraining.methods.costar import COSTAREncoder


class GroupPooling(nn.Module):
    """Weight the X/A/Y group means equally; apply dropout only to history."""

    def __init__(
        self,
        covariate_dim: int,
        treatment_dim: int,
        outcome_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.group_sizes = (covariate_dim, treatment_dim, outcome_dim)
        if any(
            isinstance(n, bool) or not isinstance(n, int) or n < 1
            for n in self.group_sizes
        ):
            raise ValueError("MIMIC requires three nonempty feature groups.")
        self.history_dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if tokens.ndim != 4 or tokens.shape[2] != sum(self.group_sizes):
            raise ValueError("Expected tokens [B,T,X+A+Y,D].")
        if valid_mask is not None:
            tokens = tokens.masked_fill(~valid_mask[:, :, None, None], 0.0)
        x, a, y = (part.mean(2) for part in tokens.split(self.group_sizes, 2))
        return {
            "covariate": x,
            "treatment": a,
            "outcome": y,
            "history": self.history_dropout((y + a + x) / 3.0),
        }


class HistoryAugmentation(nn.Module):
    """Fresh NumPy gates and CPU Torch noises, drawn in X/Y/A feature order."""

    def __init__(
        self,
        covariate_dim: int,
        treatment_dim: int,
        outcome_dim: int,
        probability: float = 0.5,
        sigma: float = 0.5,
    ) -> None:
        super().__init__()
        self.group_sizes = (covariate_dim, treatment_dim, outcome_dim)
        if min(self.group_sizes) < 1 or not 0 <= probability <= 1 or sigma < 0:
            raise ValueError("Invalid MIMIC augmentation configuration.")
        self.probability, self.sigma = probability, sigma

    def forward(
        self,
        history: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if history.ndim != 3 or history.shape[-1] != sum(self.group_sizes):
            raise ValueError("Expected history [B,T,X+A+Y].")
        if valid_mask is not None:
            history = history.masked_fill(~valid_mask[..., None], 0.0)
        x, a, y = history.split(self.group_sizes, -1)
        joined = torch.cat((x, y, a), -1)
        width = joined.shape[-1]
        # Keep the release CPU random stream even when the input is on CUDA.
        if np.random.random() <= self.probability:
            factor = 1.0 + self.sigma * torch.randn(width, device="cpu")
            joined = joined * factor.to(history.device)
        if np.random.random() <= self.probability:
            offset = self.sigma * torch.randn(width, device="cpu")
            joined = joined + offset.to(history.device)
        if np.random.random() <= self.probability:
            noise = self.sigma * torch.randn(joined.shape, device="cpu")
            joined = joined + noise.to(history.device)
        nx, na, ny = self.group_sizes
        x, y, a = joined.split((nx, ny, na), -1)
        return torch.cat((x, a, y), -1)


class GitHubCOSTAR(nn.Module):
    """Run the repository encoder with release pooling, loss scaling and forward EMA."""

    def __init__(self, configuration: Mapping[str, Any]) -> None:
        super().__init__()
        self.hparams = dict(configuration)
        names = (
            "covariate_dim",
            "treatment_dim",
            "outcome_dim",
            "static_dim",
            "d_model",
            "num_heads",
            "num_layers",
            "max_relative_distance",
            "ffn_hidden_dim",
            "dropout",
        )
        encoder_args = {name: configuration[name] for name in names}
        self.encoder = COSTAREncoder(**encoder_args)
        # A fresh second allocation preserves the released construction order.
        self.momentum_encoder = COSTAREncoder(**encoder_args)
        self.momentum_encoder.load_state_dict(self.encoder.state_dict(), strict=True)
        self.momentum_encoder.requires_grad_(False)
        groups = tuple(configuration[n] for n in names[:3])
        self.pooling = GroupPooling(*groups, configuration["dropout"])
        self.momentum_pooling = GroupPooling(*groups, configuration["dropout"])
        width = configuration["d_model"]
        hidden = configuration.get("predictor_hidden_dim", 2 * width)
        self.predictor = nn.Sequential(
            nn.Linear(width, hidden),
            nn.ReLU(),
            nn.Linear(hidden, width),
        )
        strengths = tuple(
            configuration.get(n, 0.5) for n in ("scale_std", "shift_std", "jitter_std")
        )
        if len(set(strengths)) != 1:
            raise ValueError("The GitHub profile uses one augmentation sigma.")
        self.augmentation = HistoryAugmentation(
            *groups,
            configuration.get("augmentation_probability", 0.5),
            strengths[0],
        )
        self.costar_loss = COSTARLoss(configuration.get("temperature", 1.0))
        self.momentum = float(configuration.get("momentum", 0.99))
        self.lr = float(configuration.get("lr", 0.001))
        self.weight_decay = float(configuration.get("weight_decay", 0.0))
        if not math.isfinite(self.momentum) or not 0 <= self.momentum < 1:
            raise ValueError("Invalid EMA momentum.")
        self.register_buffer("ema_updates", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update_momentum_encoder(self) -> None:
        """Update the teacher immediately before encoding its comparison keys."""
        for teacher, online in zip(
            self.momentum_encoder.parameters(),
            self.encoder.parameters(),
            strict=True,
        ):
            teacher.copy_(self.momentum * teacher + (1.0 - self.momentum) * online)
        for teacher, online in zip(
            self.momentum_encoder.buffers(),
            self.encoder.buffers(),
            strict=True,
        ):
            teacher.copy_(online)
        self.ema_updates.add_(1)

    @staticmethod
    def _codes(
        encoder: nn.Module,
        pooling: nn.Module,
        history: torch.Tensor,
        static: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        tokens, _ = encoder(history, static, mask)
        return {
            name: value.flatten(0, 1) if mask is None else value[mask]
            for name, value in pooling(tokens, mask).items()
        }

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        stage: str = "fit",
    ) -> dict[str, torch.Tensor]:
        """Compute stochastic SSL; eval/no_grad still updates teacher EMA once."""
        if stage not in {"fit", "train", "validate", "val", "test"}:
            raise ValueError("Unsupported SSL stage.")
        history, static = batch["dynamic_history"], batch["static_features"]
        mask = batch.get("valid_mask")
        first = self.augmentation(history, mask)
        second = self.augmentation(history, mask)
        codes1 = self._codes(self.encoder, self.pooling, first, static, mask)
        predictions1 = {name: self.predictor(value) for name, value in codes1.items()}
        codes2 = self._codes(self.encoder, self.pooling, second, static, mask)
        predictions2 = {name: self.predictor(value) for name, value in codes2.items()}
        with torch.no_grad():
            self.update_momentum_encoder()
            keys1 = self._codes(
                self.momentum_encoder,
                self.momentum_pooling,
                first,
                static,
                mask,
            )
            keys2 = self._codes(
                self.momentum_encoder,
                self.momentum_pooling,
                second,
                static,
                mask,
            )
        factor = 2.0 * self.costar_loss.temperature
        losses = {
            name: value * factor
            for name, value in self.costar_loss(
                predictions1,
                predictions2,
                keys1,
                keys2,
            ).items()
        }
        lengths = (
            mask.sum(1, dtype=torch.long)
            if mask is not None
            else torch.full(
                (history.shape[0],),
                history.shape[1],
                device=history.device,
                dtype=torch.long,
            )
        )
        return {
            **losses,
            "embedding": codes1["history"][lengths.cumsum(0) - 1],
            "prediction": torch.cat((predictions1["history"], predictions2["history"])),
            "target": torch.cat((keys2["history"], keys1["history"])),
        }

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Create Adam with no EMA optimizer hook."""
        return torch.optim.Adam(
            [*self.encoder.parameters(), *self.predictor.parameters()],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )


def base_model(configuration: Mapping[str, Any]) -> GitHubCOSTAR:
    """Build the experiment SSL model without importing author algorithms."""
    return GitHubCOSTAR(configuration)


class OutcomeHead(nn.Module):
    """Add history, instantaneous treatment and causal preceding-plan codes."""

    def __init__(
        self,
        d_model: int = 36,
        treatment_dim: int = 3,
        outcome_dim: int = 2,
        max_horizon: int = 11,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(d_model, treatment_dim, outcome_dim, max_horizon, hidden_dim) < 1:
            raise ValueError("Outcome dimensions must be positive.")
        if dropout != 0:
            raise ValueError("The released outcome MLP has no extra dropout.")
        self.d_model, self.max_horizon = d_model, max_horizon
        self.treatment_dim, self.outcome_dim = treatment_dim, outcome_dim
        self.instantaneous_treatment = nn.Linear(treatment_dim, d_model)
        self.preceding_treatments = nn.Conv1d(treatment_dim, d_model, max_horizon)
        self.predictor = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, outcome_dim),
        )

    def forward(
        self,
        history: torch.Tensor,
        future_treatments: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if history.ndim != 3 or future_treatments.ndim != 4:
            raise ValueError("Need history [B,T,D] and plans [B,T,H,A].")
        batch, times, horizons, treatments = future_treatments.shape
        if (
            history.shape != (batch, times, self.d_model)
            or treatments != self.treatment_dim
            or not 1 <= horizons <= self.max_horizon
        ):
            raise ValueError("Outcome history/plan dimensions disagree.")
        if target_mask is None:
            target_mask = torch.ones(
                (batch, times, horizons),
                dtype=torch.bool,
                device=history.device,
            )
        if target_mask.dtype != torch.bool or target_mask.shape != (
            batch,
            times,
            horizons,
        ):
            raise ValueError("Need boolean target_mask [B,T,H].")
        history = torch.where(target_mask.any(-1, keepdim=True), history, 0.0)
        plan = torch.where(target_mask[..., None], future_treatments, 0.0)
        # h=0 has no preceding planned treatment; h>0 excludes its current A.
        preceding = torch.cat((torch.zeros_like(plan[:, :, :1]), plan[:, :, :-1]), 2)
        preceding = preceding.reshape(batch * times, horizons, treatments).transpose(
            1, 2
        )
        prefix = self.preceding_treatments(F.pad(preceding, (self.max_horizon - 1, 0)))
        prefix = prefix.transpose(1, 2).reshape(batch, times, horizons, self.d_model)
        current = self.instantaneous_treatment(plan)
        prediction = self.predictor(history[:, :, None] + prefix + current)
        return torch.where(target_mask[..., None], prediction, 0.0)


class OutcomeModel(nn.Module):
    """Attach the released additive head to the repository encoder."""

    def __init__(
        self,
        encoder: COSTAREncoder,
        pooling: GroupPooling,
        d_model: int = 36,
        treatment_dim: int = 3,
        outcome_dim: int = 2,
        max_horizon: int = 11,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(encoder, COSTAREncoder) or not isinstance(
            pooling, GroupPooling
        ):
            raise TypeError("Reuse the repository encoder and GitHub group pooling.")
        self.encoder, self.pooling = encoder, pooling
        self.head = OutcomeHead(
            d_model,
            treatment_dim,
            outcome_dim,
            max_horizon,
            hidden_dim,
            dropout,
        )

    def encode_history(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Encode only observed history; return all causal origin codes."""
        mask = batch.get("valid_mask")
        tokens, _ = self.encoder(
            batch["dynamic_history"], batch["static_features"], mask
        )
        return self.pooling(tokens, mask)["history"]

    def predict_from_history(
        self,
        history: torch.Tensor,
        future_treatments: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if history.ndim == 2:
            return self.head(
                history[:, None],
                future_treatments[:, None],
                None if target_mask is None else target_mask[:, None],
            )[:, 0]
        return self.head(history, future_treatments, target_mask)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.predict_from_history(
            self.encode_history(batch),
            batch["future_treatments"],
            batch.get("target_mask"),
        )


def _loss_inputs(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        prediction.ndim != 4
        or target.shape != prediction.shape
        or prediction.dtype != torch.float32
        or target.dtype != torch.float32
        or mask.dtype != torch.bool
        or mask.shape != prediction.shape[:-1]
        or target.device != prediction.device
        or mask.device != prediction.device
    ):
        raise ValueError(
            "Require float32 predictions/targets [B,T,H,Y], bool mask [B,T,H]."
        )
    if not bool(mask.any()):
        raise ValueError("A supervised batch needs at least one valid origin/horizon.")
    safe_prediction = torch.where(mask[..., None], prediction, 0.0)
    safe_target = torch.where(mask[..., None], target, 0.0)
    if not bool(
        torch.isfinite(safe_prediction).all() & torch.isfinite(safe_target).all()
    ):
        raise ValueError("Available predictions and targets must be finite.")
    return safe_prediction, safe_target


def mean_one_inverse_weights(horizons: int, device: torch.device | str) -> torch.Tensor:
    """Return released float32 inverse horizon weights with mean one."""
    weights = torch.arange(
        1, horizons + 1, dtype=torch.float32, device=device
    ).reciprocal()
    return weights * (horizons / weights.sum())


def _pooled_loss_terms(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction, target = _loss_inputs(prediction, target, mask)
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        weights = mean_one_inverse_weights(prediction.shape[2], prediction.device)
        squared = F.mse_loss(prediction, target, reduction="none")
        weighted = torch.einsum("bthy,h->bthy", squared, weights)
        numerator = (weighted * mask[..., None]).sum()
        count = mask.sum()
        return numerator / count, numerator, count, weights


def weighted_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Actual float32 pooled loss; outcomes sum and denominator never divides by Y."""
    return _pooled_loss_terms(prediction, target, mask)[0]


@torch.no_grad()
def loss_statistics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return diagnostic float64 SSE[H,Y] and int64 valid-origin counts[H]."""
    prediction, target = _loss_inputs(prediction, target, mask)
    errors = prediction.double() - target.double()
    return errors.square().sum((0, 1)).cpu(), mask.sum((0, 1)).cpu()


@torch.no_grad()
def batch_loss_proof(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, Any]:
    """Save actual batch-loss arithmetic plus independent error totals for replay."""
    loss, numerator, count, weights = _pooled_loss_terms(prediction, target, mask)
    sse, counts = loss_statistics(prediction, target, mask)
    return {
        "batch_size": prediction.shape[0],
        "patients": prediction.shape[0],
        "release_validation_mse": float(loss),
        "loss": float(loss),
        "weighted_squared_error_sum_float32": float(numerator),
        "valid_origin_horizon_count": int(count),
        "squared_error_sum_per_horizon_and_outcome": sse.tolist(),
        "target_count_per_horizon": counts.tolist(),
        "horizon_weights_mean_one": weights.cpu().tolist(),
        "prediction_dtype": str(prediction.dtype),
    }
