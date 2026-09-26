"""Framework-neutral online SetGRPO update construction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .calibration import joint_coefficients


CoefficientProvider = Callable[[str, Sequence[float], Mapping[str, Any] | None], Sequence[float] | None]


@dataclass(frozen=True)
class SetGRPOUpdate:
    coefficients: tuple[float, ...]
    accepted_groups: int
    total_groups: int

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_groups / self.total_groups if self.total_groups else 0.0


def build_update(scores: Sequence[float], group_ids: Sequence[str], *, rollout_size: int, provider: CoefficientProvider | None = None, metadata: Sequence[Mapping[str, Any] | None] | None = None, allow_uncertified_proxy: bool = False) -> SetGRPOUpdate:
    if rollout_size <= 0 or len(scores) != len(group_ids):
        raise ValueError("scores, group_ids, and rollout_size are inconsistent")
    if metadata is not None and len(metadata) != len(scores):
        raise ValueError("metadata must match scores")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        groups[str(group_id)].append(index)
    invalid = {key: len(value) for key, value in groups.items() if len(value) != rollout_size}
    if invalid:
        raise ValueError(f"each group must contain rollout_size responses: {invalid}")
    output = np.zeros(len(scores), dtype=np.float64)
    accepted = 0
    if provider is None and not allow_uncertified_proxy:
        raise ValueError("a certified coefficient provider is required")
    values = np.asarray(scores, dtype=np.float64)
    for group_id, indices in groups.items():
        group_metadata = None if metadata is None else metadata[indices[0]]
        coefficients = joint_coefficients(values[indices]) if provider is None else provider(group_id, values[indices].tolist(), group_metadata)
        if coefficients is not None:
            coefficients = np.asarray(coefficients, dtype=np.float64)
            if coefficients.shape != (rollout_size,) or not np.all(np.isfinite(coefficients)):
                raise ValueError(f"provider returned invalid coefficients for {group_id}")
            output[indices] = coefficients
            accepted += 1
    return SetGRPOUpdate(tuple(float(value) for value in output), accepted, len(groups))
