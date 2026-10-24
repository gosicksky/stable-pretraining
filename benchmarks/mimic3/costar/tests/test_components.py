"""CPU semantic checks for the independent GitHub MIMIC components."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from benchmarks.mimic3.costar.common import state_digest, tensor_state
from benchmarks.mimic3.costar.components import (
    GroupPooling,
    HistoryAugmentation,
    OutcomeHead,
    OutcomeModel,
    base_model,
    batch_loss_proof,
    loss_statistics,
    weighted_loss,
)


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _restore_random_streams() -> Iterator[None]:
    numpy_state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        try:
            yield
        finally:
            np.random.set_state(numpy_state)


def config() -> dict:
    return {
        "covariate_dim": 2,
        "treatment_dim": 1,
        "outcome_dim": 2,
        "static_dim": 1,
        "d_model": 6,
        "num_heads": 2,
        "num_layers": 1,
        "max_relative_distance": 2,
        "ffn_hidden_dim": 12,
        "predictor_hidden_dim": 12,
        "dropout": 0.0,
        "temperature": 1.0,
        "momentum": 0.5,
        "lr": 0.001,
        "weight_decay": 0.0,
        "scale_std": 0.5,
        "shift_std": 0.5,
        "jitter_std": 0.5,
        "augmentation_probability": 0.5,
    }


def batch() -> dict[str, torch.Tensor]:
    return {
        "dynamic_history": torch.randn(2, 3, 5),
        "static_features": torch.randn(2, 1),
        "valid_mask": torch.tensor([[True, True, True], [True, True, False]]),
    }


def test_group_pool_weights_components_equally_and_excludes_padding():
    pooling = GroupPooling(2, 1, 2, dropout=0)
    tokens = torch.tensor([1, 3, 8, 10, 14], dtype=torch.float32).view(1, 1, 5, 1)
    tokens = tokens.expand(1, 2, 5, 3).clone()
    tokens[:, 1] = float("nan")
    output = pooling(tokens, torch.tensor([[True, False]]))
    assert torch.equal(output["history"][0, 0], torch.full((3,), 22 / 3))
    assert not torch.isclose(output["history"][0, 0, 0], torch.tensor(36 / 5))
    assert all(
        torch.equal(value[:, 1], torch.zeros_like(value[:, 1]))
        for value in output.values()
    )


def test_augmentation_affine_sharing_feature_order_and_cpu_noise(monkeypatch):
    gates = iter([1.0, 0.0, 1.0])  # Shift only.
    monkeypatch.setattr(np.random, "random", lambda: next(gates))
    devices = []

    def known_noise(size, *, device):
        devices.append(device)
        return torch.arange(size, dtype=torch.float32)

    monkeypatch.setattr(torch, "randn", known_noise)
    augmentation = HistoryAugmentation(2, 1, 2, probability=0.5, sigma=0.5)
    output = augmentation(torch.zeros(2, 3, 5))
    # CPU draw order X0,X1,Y0,Y1,A0 must return public X0,X1,A0,Y0,Y1.
    expected = torch.tensor([0.0, 0.5, 2.0, 1.0, 1.5])
    assert torch.equal(output, expected.expand_as(output))
    assert devices == ["cpu"]


def test_augmentation_eval_is_fresh_and_advances_both_random_streams():
    np.random.seed(17)
    torch.manual_seed(17)
    augmentation = HistoryAugmentation(2, 1, 2, probability=1.0).eval()
    history = torch.ones(2, 3, 5)
    first = augmentation(history)
    numpy_before = np.random.get_state()[1].copy(), np.random.get_state()[2]
    torch_before = torch.get_rng_state().clone()
    second = augmentation(history)
    assert not torch.equal(first, second)
    assert (
        not np.array_equal(numpy_before[0], np.random.get_state()[1])
        or numpy_before[1] != np.random.get_state()[2]
    )
    assert not torch.equal(torch_before, torch.get_rng_state())


def test_pooled_loss_uses_ragged_origin_denominator_and_sums_outcomes():
    prediction = torch.tensor(
        [
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
            [[[2.0, 1.0], [float("nan"), float("nan")], [float("nan"), float("nan")]]],
        ]
    )
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[True, True, True]], [[True, False, False]]])
    loss = weighted_loss(prediction, target, mask)
    # H=3 weights = [18/11,9/11,6/11], SSE_h = [10,25,61], count=4.
    expected = (10 * 18 + 25 * 9 + 61 * 6) / 11 / 4
    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(expected, rel=1e-6)
    sse, counts = loss_statistics(prediction, target, mask)
    assert torch.equal(
        sse, torch.tensor([[5, 5], [9, 16], [25, 36]], dtype=torch.float64)
    )
    assert counts.tolist() == [2, 1, 1]
    proof = batch_loss_proof(prediction, target, mask)
    assert proof["loss"] == loss.item() == proof["release_validation_mse"]
    assert proof["valid_origin_horizon_count"] == 4
    assert proof["batch_size"] == proof["patients"] == 2
    reconstructed = (
        torch.tensor(proof["weighted_squared_error_sum_float32"], dtype=torch.float32)
        / 4
    ).item()
    assert reconstructed == loss.item()


def test_additive_decoder_is_nonautoregressive_and_plan_causal():
    torch.manual_seed(31)
    head = OutcomeHead(6, 1, 2, max_horizon=4, hidden_dim=8).eval()
    assert head.predictor[0].in_features == 6
    history = torch.randn(2, 3, 6)
    plan = torch.randn(2, 3, 4, 1)
    expected = head(history, plan)
    changed = plan.clone()
    changed[:, :, 2:] += 50
    actual = head(history, changed)
    torch.testing.assert_close(actual[:, :, :2], expected[:, :, :2], rtol=0, atol=0)
    assert expected.shape == (2, 3, 4, 2)
    expected.square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()
    )


def test_ssl_all_valid_prefixes_and_prekey_ema_in_train_and_validation():
    torch.manual_seed(19)
    np.random.seed(19)
    model = base_model(config()).eval()
    initial = {
        n: p.detach().clone() for n, p in model.momentum_encoder.named_parameters()
    }
    with torch.no_grad():
        for value in model.encoder.parameters():
            value.add_(0.125)
    before = {n: p.detach().clone() for n, p in model.encoder.named_parameters()}
    observed = []

    def teacher_hook(module, _args):
        observed.append({n: p.detach().clone() for n, p in module.named_parameters()})

    hook = model.momentum_encoder.register_forward_pre_hook(teacher_hook)
    data = batch()
    result = model(data)
    hook.remove()
    assert len(observed) == 2
    for name in initial:
        expected = 0.5 * initial[name] + 0.5 * before[name]
        assert torch.equal(observed[0][name], expected)
        assert torch.equal(observed[1][name], expected)
    assert int(model.ema_updates) == 1
    assert result["prediction"].shape == result["target"].shape == (10, 6)
    assert result["embedding"].shape == (2, 6)
    assert not result["target"].requires_grad
    result["loss"].backward()
    assert all(p.grad is None for p in model.momentum_encoder.parameters())
    optimizer = model.configure_optimizers()
    optimizer.step()
    assert int(model.ema_updates) == 1  # No optimizer post-step EMA.
    online_before = {n: p.detach().clone() for n, p in model.encoder.named_parameters()}
    with torch.no_grad():
        model(data, stage="val")
    assert int(model.ema_updates) == 2
    assert all(
        torch.equal(online_before[n], p) for n, p in model.encoder.named_parameters()
    )
    model.train()
    assert model.momentum_encoder.training and model.momentum_pooling.training
    model.eval()
    assert not model.momentum_encoder.training and not model.momentum_pooling.training


def test_ssl_matches_two_temperature_scaled_component_objective(monkeypatch):
    torch.manual_seed(5)
    model = base_model(config()).eval()
    monkeypatch.setattr(model.augmentation, "forward", lambda value, mask: value)
    data = batch()
    result = model(data)
    # Identical views and equal student/teacher initialization make both
    # directions equal; the specification supplies factors2directions*2T.
    codes = model._codes(
        model.encoder,
        model.pooling,
        data["dynamic_history"],
        data["static_features"],
        data["valid_mask"],
    )
    losses = {}
    for name, value in codes.items():
        queries = F.normalize(model.predictor(value), dim=1)
        keys = F.normalize(value, dim=1)
        losses[name] = F.cross_entropy(queries @ keys.T, torch.arange(len(value))) * 4
    expected = (
        losses["history"]
        + sum(losses[n] for n in ("covariate", "treatment", "outcome")) / 3
    )
    torch.testing.assert_close(result["loss"], expected)


def test_outcome_model_reuses_repository_encoder_and_supports_cached_cf_endpoints():
    model = base_model(config())
    outcome = OutcomeModel(model.encoder, model.pooling, 6, 1, 2, 3, 8).eval()
    assert outcome.encoder is model.encoder and outcome.pooling is model.pooling
    data = batch()
    data["future_treatments"] = torch.randn(2, 3, 3, 1)
    data["target_mask"] = data["valid_mask"][..., None].expand(-1, -1, 3)
    prediction = outcome(data)
    histories = outcome.encode_history(data)
    rows = torch.arange(2)
    origins = torch.tensor([2, 1])
    endpoints = outcome.predict_from_history(
        histories[rows, origins],
        data["future_treatments"][rows, origins],
    )
    torch.testing.assert_close(endpoints, prediction[rows, origins])


def test_complete_source_best_checkpoint_initializes_target_model(
    tmp_path: Path,
) -> None:
    """Preserve the decoder as well as the encoder across the domain handoff."""
    torch.manual_seed(3017)
    source_base = base_model(config())
    source = OutcomeModel(
        source_base.encoder, source_base.pooling, 6, 1, 2, 3, 8
    ).eval()
    with torch.no_grad():
        source.head.predictor[-1].bias.add_(2.0)
    source_state = tensor_state(source)
    source_digest = state_digest(source_state)
    data = batch()
    data["future_treatments"] = torch.randn(2, 3, 3, 1)
    data["target_mask"] = data["valid_mask"][..., None].expand(-1, -1, 3)
    with torch.no_grad():
        expected = source(data)

    checkpoint = tmp_path / "source_best.pt"
    torch.save({"state_dict": source_state, "state_sha256": source_digest}, checkpoint)
    torch.manual_seed(4017)
    target_base = base_model(config())
    target = OutcomeModel(
        target_base.encoder, target_base.pooling, 6, 1, 2, 3, 8
    ).eval()
    target.encoder.load_state_dict(source.encoder.state_dict(), strict=True)
    assert state_digest(tensor_state(target)) != source_digest
    with torch.no_grad():
        assert not torch.equal(target(data), expected)

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    target.load_state_dict(saved["state_dict"], strict=True)
    target_state = tensor_state(target)
    assert set(target_state) == set(source_state)
    assert any(key.startswith("encoder.") for key in target_state)
    assert any(key.startswith("head.") for key in target_state)
    assert state_digest(target_state) == saved["state_sha256"] == source_digest
    for name, tensor in source_state.items():
        assert target_state[name].dtype == tensor.dtype
        torch.testing.assert_close(target_state[name], tensor, rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(target(data), expected, rtol=0, atol=0)
