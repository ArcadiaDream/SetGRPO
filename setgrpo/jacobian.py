"""Optional projected response-Jacobian extraction utilities."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np


def count_sketch_gradients(named_gradients: Iterable[tuple[str, Any]], *, projection_dim: int, seed: int = 0) -> Any:
    """Project parameter gradients with a deterministic signed CountSketch.

    PyTorch is imported lazily so the core package remains NumPy-only.
    """

    import torch

    if projection_dim < 1:
        raise ValueError("projection_dim must be positive")
    values = [(name, gradient) for name, gradient in named_gradients if gradient is not None]
    if not values:
        raise ValueError("at least one gradient is required")
    device = values[0][1].device
    output = torch.zeros(projection_dim, device=device, dtype=torch.float32)
    for name, gradient in values:
        flat = gradient.detach().reshape(-1).to(dtype=torch.float32)
        generator = torch.Generator(device=device)
        generator.manual_seed((hash((name, seed)) & ((1 << 63) - 1)))
        buckets = torch.randint(projection_dim, (flat.numel(),), generator=generator, device=device)
        signs = torch.randint(0, 2, (flat.numel(),), generator=generator, device=device).float().mul_(2).sub_(1)
        output.scatter_add_(0, buckets, flat * signs)
    return output


def validate_response_jacobian(jacobian: Sequence[Sequence[float]], *, projection_dim: int, rollout_size: int) -> np.ndarray:
    """Validate a projected response Jacobian with shape ``[d, rollout_size]``."""

    matrix = np.asarray(jacobian, dtype=np.float64)
    if matrix.shape != (projection_dim, rollout_size) or not np.all(np.isfinite(matrix)):
        raise ValueError("jacobian has an invalid shape or contains non-finite values")
    if np.any(np.linalg.norm(matrix, axis=0) == 0):
        raise ValueError("every response Jacobian column must be non-zero")
    return matrix
