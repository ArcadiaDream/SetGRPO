"""Joint reward-set calibration for SetGRPO."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Sequence

import numpy as np


class CalibrationError(ValueError):
    """Raised when a coefficient set cannot be constructed."""


Array = np.ndarray


def _vector(value: Sequence[float] | Array, name: str) -> Array:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise CalibrationError(f"{name} must be a non-empty finite vector")
    return result


def _matrix(value: Sequence[Sequence[float]] | Array, name: str) -> Array:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] != result.shape[1] or not np.all(np.isfinite(result)):
        raise CalibrationError(f"{name} must be a finite square matrix")
    return result


def _check_spd(value: Array, name: str) -> None:
    try:
        np.linalg.cholesky(value)
    except np.linalg.LinAlgError as exc:
        raise CalibrationError(f"{name} must be positive definite") from exc


def joint_coefficients(scores: Sequence[float] | Array, *, atol: float = 1e-12) -> Array:
    """Normalize one GRPO reward group into its coefficient vector."""

    rewards = _vector(scores, "scores")
    if rewards.size < 2:
        raise CalibrationError("a reward group needs at least two responses")
    centered = rewards - rewards.mean()
    scale = float(np.sqrt(np.dot(centered, centered) / rewards.size))
    if scale <= atol:
        raise CalibrationError("reward group standard deviation is zero")
    return centered / scale


def standardization_jacobian(scores: Sequence[float] | Array, *, atol: float = 1e-12) -> Array:
    """Return the analytic Jacobian of ``joint_coefficients``."""

    rewards = _vector(scores, "scores")
    centered = rewards - rewards.mean()
    scale = float(np.sqrt(np.dot(centered, centered) / rewards.size))
    if scale <= atol:
        raise CalibrationError("reward group standard deviation is zero")
    return (np.eye(rewards.size) - np.ones((rewards.size, rewards.size)) / rewards.size - np.outer(centered / scale, centered / scale) / rewards.size) / scale


def split_conformal_radius(residual_radii: Sequence[float], *, alpha: float) -> float:
    """Compute the finite-sample split-conformal radius."""

    if not 0 < alpha < 1:
        raise CalibrationError("alpha must be in (0, 1)")
    values = _vector(residual_radii, "residual_radii")
    if np.any(values < 0):
        raise CalibrationError("residual_radii must be non-negative")
    rank = ceil((values.size + 1) * (1 - alpha))
    return float("inf") if rank > values.size else float(np.sort(values)[rank - 1])


def mahalanobis_radius(reference: Sequence[float], estimate: Sequence[float], shape: Sequence[Sequence[float]]) -> float:
    """Return the joint coefficient residual radius."""

    ref = _vector(reference, "reference")
    est = _vector(estimate, "estimate")
    covariance = _matrix(shape, "shape")
    if ref.shape != est.shape or covariance.shape != (ref.size, ref.size):
        raise CalibrationError("coefficient and shape dimensions do not match")
    _check_spd(covariance, "shape")
    delta = ref - est
    return float(np.sqrt(max(0.0, delta @ np.linalg.solve(covariance, delta))))


def covariance_shape(jacobian: Sequence[Sequence[float]], variance: Sequence[Sequence[float]], *, ridge: float = 1e-8) -> Array:
    """Construct ``J V J^T + ridge I`` for a joint coefficient set."""

    matrix = _matrix(jacobian, "jacobian")
    score_covariance = _matrix(variance, "variance")
    if matrix.shape[1] != score_covariance.shape[0] or ridge < 0:
        raise CalibrationError("jacobian, variance, or ridge has an invalid shape/value")
    shape = matrix @ score_covariance @ matrix.T + ridge * np.eye(matrix.shape[0])
    _check_spd(shape, "shape")
    return shape


@dataclass(frozen=True)
class CoefficientSet:
    """Ellipsoidal joint uncertainty set ``{a:(a-c)^T S^-1(a-c)<=q^2}``."""

    center: Array
    shape: Array
    radius: float

    def __post_init__(self) -> None:
        center = _vector(self.center, "center")
        shape = _matrix(self.shape, "shape")
        if shape.shape != (center.size, center.size) or not np.isfinite(self.radius) or self.radius < 0:
            raise CalibrationError("invalid coefficient set dimensions or radius")
        _check_spd(shape, "shape")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "shape", shape)


class JointRewardSetCalibration:
    """Build a joint coefficient set from paired reward observations."""

    def __init__(self, *, alpha: float = 0.1, ridge: float = 1e-8) -> None:
        self.alpha = alpha
        self.ridge = ridge

    def fit(self, estimated_scores: Sequence[float], reference_scores: Sequence[float], score_variance: Sequence[Sequence[float]]) -> CoefficientSet:
        estimate = joint_coefficients(estimated_scores)
        reference = joint_coefficients(reference_scores)
        jacobian = standardization_jacobian(estimated_scores)
        shape = covariance_shape(jacobian, score_variance, ridge=self.ridge)
        radius = mahalanobis_radius(reference, estimate, shape)
        return CoefficientSet(estimate, shape, radius)
