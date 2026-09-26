"""Geometry-Guided Safe Recovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from .calibration import CoefficientSet
from .geometry import LocalPolicyGeometry


@dataclass(frozen=True)
class RecoveryResult:
    accepted: bool
    eta: float
    coefficients: np.ndarray | None
    update: np.ndarray | None
    full_margin: float | None
    reward_margin: float | None


def support_margin(direction: Sequence[float], coefficient_set: CoefficientSet, geometry: LocalPolicyGeometry, *, numerical_tolerance: float = 0.0) -> float:
    vector = np.asarray(direction, dtype=np.float64)
    nominal = float(vector @ geometry.update(coefficient_set.center))
    pulled = geometry.pullback(vector)
    penalty = coefficient_set.radius * float(np.sqrt(max(0.0, pulled @ coefficient_set.shape @ pulled)))
    return nominal - penalty - numerical_tolerance


def safe_anchor(coefficient_set: CoefficientSet, geometry: LocalPolicyGeometry) -> tuple[np.ndarray, np.ndarray]:
    """Project the coefficient set toward the smallest local update."""

    inverse = np.linalg.inv(coefficient_set.shape)
    center = coefficient_set.center

    def objective(values: np.ndarray) -> float:
        update = geometry.update(values)
        return 0.5 * float(update @ update)

    def gradient(values: np.ndarray) -> np.ndarray:
        update = geometry.update(values)
        return geometry.response_jacobian.T @ geometry.projection.T @ update

    def constraint(values: np.ndarray) -> float:
        delta = values - center
        return float(coefficient_set.radius**2 - delta @ inverse @ delta)

    result = minimize(objective, center.copy(), jac=gradient, constraints={"type": "ineq", "fun": constraint}, method="SLSQP")
    anchor = np.asarray(result.x if result.success else center, dtype=np.float64)
    return anchor, geometry.update(anchor)


class GeometryGuidedSafeRecovery:
    """Recover the largest certified interpolation toward nominal GRPO."""

    def __init__(self, coefficient_set: CoefficientSet, geometry: LocalPolicyGeometry, reward_geometry: LocalPolicyGeometry | None = None, *, grid_size: int = 257) -> None:
        self.coefficient_set = coefficient_set
        self.geometry = geometry
        self.reward_geometry = reward_geometry or geometry
        self.grid_size = grid_size

    def recover(self, *, threshold: float, numerical_tolerance: float = 0.0, require_reward_progress: bool = True) -> RecoveryResult:
        anchor, _ = safe_anchor(self.coefficient_set, self.geometry)
        best = None
        for eta in np.linspace(0.0, 1.0, self.grid_size):
            coefficients = (1.0 - eta) * anchor + eta * self.coefficient_set.center
            update = self.geometry.update(coefficients)
            reward_update = self.reward_geometry.update(coefficients)
            if np.linalg.norm(update) <= 1e-12:
                continue
            full = support_margin(update, self.coefficient_set, self.geometry, numerical_tolerance=numerical_tolerance)
            reward = support_margin(reward_update, self.coefficient_set, self.reward_geometry, numerical_tolerance=numerical_tolerance)
            if full >= threshold and (reward > 0.0 or not require_reward_progress):
                best = (float(eta), coefficients, update, full, reward)
        if best is None:
            return RecoveryResult(False, 0.0, None, None, None, None)
        eta, coefficients, update, full, reward = best
        return RecoveryResult(True, eta, coefficients, update, full, reward)
