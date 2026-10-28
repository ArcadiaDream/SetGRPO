"""Selective Risk-Gated Optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from scipy.stats import beta


@dataclass(frozen=True)
class RiskCalibrationResult:
    selected_threshold: float | None
    threshold_rows: tuple[dict[str, float | int | bool], ...]


def clopper_pearson_upper_bound(failures: int, trials: int, *, confidence: float) -> float:
    if trials <= 0:
        return 1.0
    if failures == trials:
        return 1.0
    return float(beta.ppf(confidence, failures + 1, trials - failures))


class SelectiveRiskGatedOptimization:
    """Select a threshold under a simultaneous direction-reversal bound."""

    def __init__(self, *, beta_target: float, gamma: float) -> None:
        if not 0 < beta_target < 1 or not 0 < gamma < 1:
            raise ValueError("beta_target and gamma must be in (0, 1)")
        self.beta_target = beta_target
        self.gamma = gamma

    def select(self, rows: Iterable[Mapping[str, object]], thresholds: Sequence[float]) -> RiskCalibrationResult:
        grid = tuple(float(value) for value in thresholds)
        if not grid:
            raise ValueError("thresholds must not be empty")
        rows = tuple(rows)
        confidence = 1.0 - self.gamma / len(grid)
        summaries = []
        for threshold in grid:
            accepted = [row for row in rows if float(row["threshold"]) == threshold and bool(row.get("accepted", False))]
            failures = sum(bool(row.get("reversed", False)) for row in accepted)
            upper = clopper_pearson_upper_bound(failures, len(accepted), confidence=confidence)
            summaries.append({"threshold": threshold, "trials": len(accepted), "failures": failures, "upper_bound": upper, "yield": float(sum(float(row.get("yield", 1.0)) for row in accepted)), "selectable": len(accepted) > 0 and upper <= self.beta_target})
        selectable = [row for row in summaries if row["selectable"]]
        selected = max(selectable, key=lambda row: (float(row["yield"]), -float(row["threshold"]))) if selectable else None
        return RiskCalibrationResult(None if selected is None else float(selected["threshold"]), tuple(summaries))
