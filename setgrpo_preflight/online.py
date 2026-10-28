"""Online SetGRPO step helpers.

This module only handles the driver-side shape transformation needed between
rollout/reward computation and the trainer's advantage estimator.  Policy
geometry and certificate decisions are injected through the optional
``coefficient_provider`` so the training loop does not silently fall back to a
proxy implementation.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from .core import SetGRPOError, group_standardized_coefficients


@dataclass(frozen=True)
class OnlineSetGRPOStep:
    """Per-response coefficients and diagnostics for one rollout step."""

    coefficients: tuple[float, ...]
    group_ids: tuple[str, ...]
    accepted_groups: int
    total_groups: int

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_groups / self.total_groups if self.total_groups else 0.0


CoefficientProvider = Callable[[str, Sequence[float], Mapping[str, Any] | None], Sequence[float] | None]


def load_coefficient_provider(path: str | Path, name: str = "build_coefficients") -> CoefficientProvider:
    """Load a coefficient provider from a user-specified Python module."""
    module_path = Path(path).expanduser()
    if not module_path.is_file():
        raise FileNotFoundError(f"SetGRPO coefficient provider not found: {module_path}")
    spec = importlib.util.spec_from_file_location("setgrpo_online_provider", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load SetGRPO coefficient provider: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    provider = getattr(module, name, None)
    if not callable(provider):
        raise AttributeError(f"Provider {name!r} not found in {module_path}")
    return provider


def build_online_step(
    scores: Sequence[float],
    group_ids: Sequence[str],
    *,
    rollout_n: int,
    coefficient_provider: CoefficientProvider | None = None,
    metadata: Sequence[Mapping[str, Any] | None] | None = None,
    allow_reward_proxy: bool = False,
) -> OnlineSetGRPOStep:
    """Build one step's response coefficients in rollout order.

    ``group_ids`` must be ordered exactly like the repeated rollout batch.
    Every group must contain ``rollout_n`` responses.  A provider may return a
    certified coefficient vector for a group; returning ``None`` fail-closes
    that group to zeros.  Without a provider, centered standardized rewards
    are available only when ``allow_reward_proxy`` is explicitly enabled.
    """

    if rollout_n <= 0:
        raise SetGRPOError("rollout_n must be positive")
    if len(scores) != len(group_ids):
        raise SetGRPOError("scores and group_ids must have the same length")
    if metadata is not None and len(metadata) != len(scores):
        raise SetGRPOError("metadata must match scores length")

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        grouped[str(group_id)].append(index)
    bad = {group: len(indices) for group, indices in grouped.items() if len(indices) != rollout_n}
    if bad:
        raise SetGRPOError(f"each group must contain rollout_n responses; invalid groups: {bad}")

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise SetGRPOError("scores must be a finite one-dimensional sequence")
    output = np.zeros(values.size, dtype=np.float64)
    accepted = 0

    if coefficient_provider is None and not allow_reward_proxy:
        raise SetGRPOError(
            "online SetGRPO requires a coefficient provider; "
            "set allow_reward_proxy=true only for diagnostics"
        )

    for group_id, indices in grouped.items():
        group_scores = values[indices]
        group_meta = metadata[indices[0]] if metadata is not None else None
        if coefficient_provider is None:
            try:
                group_coefficients = group_standardized_coefficients(group_scores)
                accepted += 1
            except SetGRPOError:
                group_coefficients = np.zeros(rollout_n, dtype=np.float64)
        else:
            provided = coefficient_provider(group_id, group_scores.tolist(), group_meta)
            if provided is None:
                group_coefficients = np.zeros(rollout_n, dtype=np.float64)
            else:
                group_coefficients = np.asarray(provided, dtype=np.float64)
                if group_coefficients.shape != (rollout_n,) or not np.all(np.isfinite(group_coefficients)):
                    raise SetGRPOError(
                        f"coefficient provider returned invalid shape/values for group {group_id}"
                    )
                accepted += 1
        output[indices] = group_coefficients

    return OnlineSetGRPOStep(
        coefficients=tuple(float(value) for value in output),
        group_ids=tuple(str(value) for value in group_ids),
        accepted_groups=accepted,
        total_groups=len(grouped),
    )


def attach_online_coefficients(batch: Any, step: OnlineSetGRPOStep, *, key: str = "setgrpo_coefficients") -> Any:
    """Attach coefficients to a verl ``DataProto`` batch without copying it."""

    import torch

    if not hasattr(batch, "batch"):
        raise TypeError("batch must expose a .batch mapping")
    reference = batch.batch.get("responses")
    if reference is None or not isinstance(reference, torch.Tensor):
        raise SetGRPOError("batch must contain tensor batch['responses']")
    if reference.shape[0] != len(step.coefficients):
        raise SetGRPOError("coefficient count must match batch response count")
    batch.batch[key] = torch.as_tensor(step.coefficients, dtype=torch.float32, device=reference.device)
    return batch
