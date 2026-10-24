"""Runtime helpers shared by the portable COSTAR evaluation tools."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler


def utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    """Enforce a runtime invariant.

    Args:
        condition: Whether the invariant holds.
        message: Error message used when the invariant fails.

    Raises:
        RuntimeError: If condition is false.
    """
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    """Compute a file's SHA-256 digest.

    Args:
        path: File to read.

    Returns:
        Hexadecimal SHA-256 digest of the file bytes.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    """Read UTF-8 JSON, accepting an optional byte order mark.

    Args:
        path: JSON file to read.

    Returns:
        The decoded JSON value.
    """
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    """Write finite JSON atomically, retrying transient Windows file locks.

    Args:
        path: Destination file; missing parent directories are created.
        value: JSON-serializable value without NaN or infinity.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for attempt in range(30):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as error:
            if (
                os.name != "nt"
                or getattr(error, "winerror", None) not in (5, 32, 33)
                or attempt == 29
            ):
                raise
            time.sleep(0.05)


def save_checkpoint(path: Path, value: Any) -> None:
    """Save a PyTorch checkpoint atomically with Windows file-lock retries.

    Args:
        path: Destination file; missing parent directories are created.
        value: Checkpoint payload accepted by torch.save.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    for attempt in range(30):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as error:
            if (
                os.name != "nt"
                or getattr(error, "winerror", None) not in (5, 32, 33)
                or attempt == 29
            ):
                raise
            time.sleep(0.05)


def tensor_state(module: nn.Module) -> dict[str, torch.Tensor]:
    """Copy a module's state tensors to detached CPU storage.

    Args:
        module: Module whose parameters and buffers are copied.

    Returns:
        State dictionary containing independent CPU tensor copies.
    """
    return {
        key: value.detach().cpu().clone() for key, value in module.state_dict().items()
    }


def state_digest(state: dict[str, torch.Tensor]) -> str:
    """Hash named tensors, including their dtypes and shapes.

    Args:
        state: Named tensors to hash in sorted key order.

    Returns:
        Hexadecimal SHA-256 digest of the state.
    """
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch CPU and CUDA random generators.

    Args:
        seed: Seed shared by the random generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def numerical_environment() -> None:
    """Configure four CPU threads and deterministic PyTorch operations."""
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class RecordingSampler(RandomSampler):
    """Record the actual local dataset order without using the model RNG."""

    def __iter__(self) -> Iterator[int]:
        self.last_order = list(super().__iter__())
        yield from self.last_order


def patient_order_digest(train_loader: DataLoader) -> str:
    """Hash the most recently recorded training sample order.

    Args:
        train_loader: Loader with a RecordingSampler that has been iterated.

    Returns:
        Hexadecimal SHA-256 digest of the sampled dataset indices.
    """
    order = np.asarray(train_loader.sampler.last_order, dtype="<i8")
    return hashlib.sha256(order.tobytes()).hexdigest()


def loader(
    dataset: Any, batch_size: int, *, seed: int = 0, shuffle: bool = False
) -> DataLoader:
    """Build a loader with its own random generator and no dropped batches.

    Args:
        dataset: Dataset consumed by the loader.
        batch_size: Maximum samples in each batch.
        seed: Seed for the loader and optional shuffle sampler.
        shuffle: Whether to shuffle and record each sampled index order.

    Returns:
        A loader that reads data in the main process.
    """
    generator = torch.Generator().manual_seed(seed)
    sampler = RecordingSampler(dataset, generator=generator) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        generator=generator,
        num_workers=0,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def move(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    """Move all tensors in a batch to a device.

    Args:
        batch: Mapping of batch field names to tensors.
        device: Destination device accepted by torch.Tensor.to.

    Returns:
        A new mapping containing the transferred tensors.
    """
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def finite_gradients(module: nn.Module) -> bool:
    """Check that every existing parameter gradient is finite.

    Args:
        module: Module whose gradients are inspected.

    Returns:
        True if all existing gradients are finite, including when none exist.
    """
    return all(
        bool(torch.isfinite(p.grad).all())
        for p in module.parameters()
        if p.grad is not None
    )


def rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, and PyTorch CPU and CUDA random states.

    Returns:
        A checkpoint-compatible state mapping accepted by restore_rng.
    """
    numpy = np.random.get_state()
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy[0],
            "keys": torch.from_numpy(numpy[1].astype(np.int64)),
            "position": numpy[2],
            "has_gauss": numpy[3],
            "cached_gaussian": numpy[4],
        },
    }


def restore_rng(state: dict[str, Any]) -> None:
    """Restore the random generator states captured by rng_state.

    Args:
        state: Previously captured random generator state mapping.
    """
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state(
        (
            numpy["algorithm"],
            numpy["keys"].numpy().astype(np.uint32),
            numpy["position"],
            numpy["has_gauss"],
            numpy["cached_gaussian"],
        )
    )


def stream_digests() -> dict[str, str]:
    """Record CPU augmentation streams without drawing random values.

    Returns:
        SHA-256 digests of the PyTorch CPU and NumPy random states.
    """
    numpy = np.random.get_state()
    numpy_bytes = numpy[1].tobytes() + repr(numpy[2:]).encode()
    return {
        "torch_cpu": hashlib.sha256(
            torch.get_rng_state().numpy().tobytes()
        ).hexdigest(),
        "numpy": hashlib.sha256(numpy_bytes).hexdigest(),
    }


@contextmanager
def evaluation(module: nn.Module, *, fixed_seed: int | None = None) -> Iterator[None]:
    """Evaluate without gradients while preserving modes and PyTorch RNG state.

    Args:
        module: Module to evaluate; it must have at least one parameter.
        fixed_seed: Optional temporary seed for PyTorch CPU and, for a CUDA
            module, the current CUDA device.

    Yields:
        None while evaluation mode and disabled gradient tracking are active.

    Note:
        Restores each submodule's training flag and the PyTorch CPU RNG state.
        For CUDA modules, also preserves the current CUDA device's RNG state.
    """
    flags = {name: child.training for name, child in module.named_modules()}
    devices = [torch.cuda.current_device()] if next(module.parameters()).is_cuda else []
    try:
        with torch.random.fork_rng(devices=devices):
            if fixed_seed is not None:
                torch.random.default_generator.manual_seed(fixed_seed)
                if devices:
                    torch.cuda.manual_seed(fixed_seed)
            module.eval()
            with torch.no_grad():
                yield
    finally:
        for name, child in module.named_modules():
            child.training = flags[name]
