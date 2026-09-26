"""Minimal training-framework adapter for applying SetGRPO to GRPO batches."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .online import CoefficientProvider, build_update


def attach_coefficients(batch: Any, coefficients: Sequence[float], *, key: str = "setgrpo_coefficients") -> Any:
    """Attach coefficients to a tensor batch without depending on a trainer."""

    import torch
    if not hasattr(batch, "batch") or "responses" not in batch.batch:
        raise TypeError("batch must expose batch['responses']")
    reference = batch.batch["responses"]
    batch.batch[key] = torch.as_tensor(coefficients, dtype=torch.float32, device=reference.device)
    return batch


def prepare_grpo_batch(batch: Any, *, scores: Sequence[float], group_ids: Sequence[str], rollout_size: int, provider: CoefficientProvider, metadata: Sequence[Mapping[str, Any] | None] | None = None) -> Any:
    """Compute certified coefficients and attach them for a GRPO loss."""

    update = build_update(scores, group_ids, rollout_size=rollout_size, provider=provider, metadata=metadata)
    return attach_coefficients(batch, update.coefficients)
