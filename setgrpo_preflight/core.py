"""SetGRPO P0 math utilities.

This module intentionally implements the *local, frozen-microbatch* math from
the SetGRPO design without touching the online trainer.  The functions here are
small and auditable: group normalization sensitivity, joint conformal
coefficient sets, ellipsoid support certificates, progress recovery, and
selective-risk threshold selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist


Array = NDArray[np.float64]


class SetGRPOError(ValueError):
    """Raised when a SetGRPO certificate cannot be constructed safely."""


@dataclass(frozen=True)
class CoefficientEllipsoid:
    """Joint coefficient set ``C_x`` from Eq. (6).

    ``center`` is ``ahat_x``.  ``shape`` is the positive-definite matrix ``S_x``.
    ``radius`` is the split-conformal radius ``q_alpha``.
    """

    center: Array
    shape: Array
    radius: float

    def __post_init__(self) -> None:
        center = _vector(self.center, "center")
        shape = _matrix(self.shape, "shape")
        if shape.shape != (center.size, center.size):
            raise SetGRPOError("shape must be square with dimension matching center")
        if not np.isfinite(self.radius) or self.radius < 0:
            raise SetGRPOError("radius must be finite and non-negative")
        _assert_spd(shape, "shape")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "shape", shape)


@dataclass(frozen=True)
class LocalGeometry:
    """Frozen local affine map ``u_x(a)=P_x(G_x a+h_x)`` from Eq. (3)."""

    G: Array
    P: Array
    h: Array

    def __post_init__(self) -> None:
        G = _matrix(self.G, "G")
        P = _matrix(self.P, "P")
        h = _vector(self.h, "h")
        if G.shape[0] != h.size:
            raise SetGRPOError("G rows must match h dimension")
        if P.shape != (h.size, h.size):
            raise SetGRPOError("P must be square with dimension matching h")
        object.__setattr__(self, "G", G)
        object.__setattr__(self, "P", P)
        object.__setattr__(self, "h", h)

    def update(self, coefficients: Sequence[float] | Array) -> Array:
        """Return ``u_x(a)`` for a coefficient vector."""

        a = _vector(coefficients, "coefficients")
        if a.size != self.G.shape[1]:
            raise SetGRPOError("coefficients dimension does not match G columns")
        return self.P @ (self.G @ a + self.h)


@dataclass(frozen=True)
class RecoveryResult:
    """Result of Eq. (8a) for one frozen microbatch and threshold."""

    accepted: bool
    eta_star: float
    coefficients: Array | None
    update: Array | None
    full_margin: float | None
    reward_margin: float | None


@dataclass(frozen=True)
class SelectiveRiskResult:
    """Threshold-selection result from Eq. (8b)."""

    selected_threshold: float | None
    threshold_rows: tuple[dict[str, float | int | bool], ...]


def group_centering_matrix(group_size: int) -> Array:
    """Return ``H_G = I_G - 11^T/G`` from Eq. (2a)."""

    if group_size < 2:
        raise SetGRPOError("group_size must be at least 2")
    ones = np.ones((group_size, group_size), dtype=np.float64)
    return np.eye(group_size, dtype=np.float64) - ones / group_size


def group_standard_deviation(scores: Sequence[float] | Array, *, atol: float = 1e-12) -> float:
    """Return the population standard deviation used by Eq. (2a)."""

    r = _vector(scores, "scores")
    H = group_centering_matrix(r.size)
    centered = H @ r
    scale = float(np.sqrt(np.dot(centered, centered) / r.size))
    if scale <= atol:
        raise SetGRPOError("group standard deviation is zero or below tolerance")
    return scale


def group_standardized_coefficients(scores: Sequence[float] | Array, *, atol: float = 1e-12) -> Array:
    """Return ``z_b(r_b)=H_G r_b / s_b(r_b)`` from Eq. (2a)."""

    r = _vector(scores, "scores")
    H = group_centering_matrix(r.size)
    return (H @ r) / group_standard_deviation(r, atol=atol)


def group_standardization_jacobian(scores: Sequence[float] | Array, *, atol: float = 1e-12) -> Array:
    """Return the analytic Jacobian ``Dz_b(r_b)`` from Eq. (2b)."""

    r = _vector(scores, "scores")
    H = group_centering_matrix(r.size)
    s = group_standard_deviation(r, atol=atol)
    z = (H @ r) / s
    return (H - np.outer(z, z) / r.size) / s


def linearized_standardization_error(
    estimated_scores: Sequence[float] | Array,
    reference_scores: Sequence[float] | Array,
    *,
    atol: float = 1e-12,
) -> tuple[Array, Array]:
    """Return actual and first-order coefficient errors from Eq. (2c).

    The first element is ``z(r*) - z(rhat)``.  The second is
    ``Dz(rhat) @ (r* - rhat)``.
    """

    estimated = _vector(estimated_scores, "estimated_scores")
    reference = _vector(reference_scores, "reference_scores")
    if estimated.shape != reference.shape:
        raise SetGRPOError("estimated_scores and reference_scores must have the same shape")
    actual = group_standardized_coefficients(reference, atol=atol) - group_standardized_coefficients(
        estimated, atol=atol
    )
    linearized = group_standardization_jacobian(estimated, atol=atol) @ (reference - estimated)
    return actual, linearized


def block_diagonal_shape(
    jacobians: Sequence[Sequence[Sequence[float]] | Array],
    variances: Sequence[Sequence[Sequence[float]] | Array],
    *,
    ridge: float = 1e-8,
) -> Array:
    """Construct ``S_x`` from Eq. (5) as blockdiag ``J V J^T + lambda I``."""

    if ridge < 0 or not np.isfinite(ridge):
        raise SetGRPOError("ridge must be finite and non-negative")
    if len(jacobians) != len(variances):
        raise SetGRPOError("jacobians and variances must have the same number of blocks")
    blocks: list[Array] = []
    for idx, (jacobian, variance) in enumerate(zip(jacobians, variances)):
        J = _matrix(jacobian, f"jacobians[{idx}]")
        V = _matrix(variance, f"variances[{idx}]")
        if V.shape[0] != V.shape[1]:
            raise SetGRPOError(f"variances[{idx}] must be square")
        if J.shape[1] != V.shape[0]:
            raise SetGRPOError(f"jacobians[{idx}] columns must match variances[{idx}]")
        block = J @ V @ J.T + ridge * np.eye(J.shape[0], dtype=np.float64)
        _assert_spd(block, f"shape block {idx}")
        blocks.append(block)
    return _block_diag(blocks)


def mahalanobis_radius(
    reference_coefficients: Sequence[float] | Array,
    estimated_coefficients: Sequence[float] | Array,
    shape: Sequence[Sequence[float]] | Array,
) -> float:
    """Return ``sqrt((a*-ahat)^T S^-1 (a*-ahat))`` for conformal calibration."""

    reference = _vector(reference_coefficients, "reference_coefficients")
    estimated = _vector(estimated_coefficients, "estimated_coefficients")
    S = _matrix(shape, "shape")
    if reference.shape != estimated.shape:
        raise SetGRPOError("reference and estimated coefficients must have the same shape")
    if S.shape != (reference.size, reference.size):
        raise SetGRPOError("shape dimension must match coefficients")
    _assert_spd(S, "shape")
    delta = reference - estimated
    solved = np.linalg.solve(S, delta)
    return float(np.sqrt(max(0.0, np.dot(delta, solved))))


def split_conformal_radius(residual_radii: Sequence[float], *, alpha: float) -> float:
    """Return split-conformal quantile ``q_alpha``.

    Uses the standard finite-sample index ``ceil((n+1)(1-alpha))`` on sorted
    residuals.  If the requested rank is beyond the calibration set, the method
    fails closed by returning ``inf``.
    """

    if not 0 < alpha < 1:
        raise SetGRPOError("alpha must be in (0, 1)")
    residuals = np.asarray(tuple(residual_radii), dtype=np.float64)
    if residuals.ndim != 1 or residuals.size == 0:
        raise SetGRPOError("residual_radii must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(residuals)) or np.any(residuals < 0):
        raise SetGRPOError("residual_radii must be finite and non-negative")
    rank = ceil((residuals.size + 1) * (1 - alpha))
    if rank > residuals.size:
        return float("inf")
    return float(np.sort(residuals)[rank - 1])


def support_margin(
    direction: Sequence[float] | Array,
    ellipsoid: CoefficientEllipsoid,
    geometry: LocalGeometry,
    *,
    tau_num: float = 0.0,
) -> float:
    """Return ``psi_x(v)`` from Eq. (7a)."""

    v = _vector(direction, "direction")
    if v.size != geometry.P.shape[0]:
        raise SetGRPOError("direction dimension must match geometry output dimension")
    if tau_num < 0 or not np.isfinite(tau_num):
        raise SetGRPOError("tau_num must be finite and non-negative")
    nominal = float(np.dot(v, geometry.update(ellipsoid.center)))
    pulled_back = geometry.G.T @ (geometry.P.T @ v)
    penalty_sq = float(pulled_back.T @ ellipsoid.shape @ pulled_back)
    penalty = ellipsoid.radius * np.sqrt(max(0.0, penalty_sq))
    return nominal - penalty - tau_num


def direct_worst_case_support(
    direction: Sequence[float] | Array,
    ellipsoid: CoefficientEllipsoid,
    geometry: LocalGeometry,
) -> float:
    """Numerically solve the original constrained minimum in Eq. (7a).

    This is intended for P0-B closed-form validation, not for the training hot
    path.
    """

    v = _vector(direction, "direction")
    S_inv = np.linalg.inv(ellipsoid.shape)

    def objective(a: Array) -> float:
        return float(np.dot(v, geometry.update(a)))

    def constraint(a: Array) -> float:
        delta = a - ellipsoid.center
        return float(ellipsoid.radius**2 - delta.T @ S_inv @ delta)

    result = minimize(
        objective,
        ellipsoid.center.copy(),
        constraints={"type": "ineq", "fun": constraint},
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success:
        raise SetGRPOError(f"direct support minimization failed: {result.message}")
    return float(result.fun)


def safe_anchor(ellipsoid: CoefficientEllipsoid, geometry: LocalGeometry) -> tuple[Array, Array]:
    """Return ``(a_x^proj, d_x)`` from Eq. (7)."""

    S_inv = np.linalg.inv(ellipsoid.shape)

    def objective(a: Array) -> float:
        u = geometry.update(a)
        return 0.5 * float(np.dot(u, u))

    def gradient(a: Array) -> Array:
        u = geometry.update(a)
        return geometry.G.T @ (geometry.P.T @ u)

    def constraint(a: Array) -> float:
        delta = a - ellipsoid.center
        return float(ellipsoid.radius**2 - delta.T @ S_inv @ delta)

    def constraint_jac(a: Array) -> Array:
        return -2.0 * (S_inv @ (a - ellipsoid.center))

    result = minimize(
        objective,
        ellipsoid.center.copy(),
        jac=gradient,
        constraints={"type": "ineq", "fun": constraint, "jac": constraint_jac},
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success:
        # SLSQP can reject a numerically tiny ellipsoid boundary even though
        # the convex problem is feasible.  A radial boundary point is a
        # deterministic feasible fallback; it is conservative and keeps the
        # certificate fail-closed without aborting an entire calibration run.
        center = ellipsoid.center
        candidates = [center.copy()]
        zero = np.zeros_like(center)
        zero_delta = zero - center
        zero_feasibility = float(ellipsoid.radius**2 - zero_delta.T @ S_inv @ zero_delta)
        if zero_feasibility >= -1e-8:
            candidates.append(zero)

        center_quadratic = float(center.T @ S_inv @ center)
        if center_quadratic > 0.0 and ellipsoid.radius > 0.0:
            boundary_fraction = min(1.0, ellipsoid.radius / np.sqrt(center_quadratic))
            # Step slightly inside the boundary.  This avoids aborting on
            # ill-conditioned smoke-test ellipsoids where the exact radial
            # boundary can evaluate marginally infeasible after inversion.
            for shrink in (1.0 - 1e-10, 1.0 - 1e-8, 1.0 - 1e-6, 0.999):
                candidates.append((1.0 - boundary_fraction * shrink) * center)

        feasible: list[Array] = []
        for candidate in candidates:
            delta = candidate - center
            feasibility = float(ellipsoid.radius**2 - delta.T @ S_inv @ delta)
            if feasibility >= -1e-7:
                feasible.append(candidate)
        a_proj = min(feasible, key=objective) if feasible else center.copy()
        return a_proj, geometry.update(a_proj)
    a_proj = np.asarray(result.x, dtype=np.float64)
    return a_proj, geometry.update(a_proj)


def recover_progress(
    ellipsoid: CoefficientEllipsoid,
    full_geometry: LocalGeometry,
    reward_geometry: LocalGeometry,
    *,
    tau: float,
    tau_num: float = 0.0,
    tau_num_reward: float = 0.0,
    grid_size: int = 257,
    margin_tolerance: float = 0.0,
    require_reward_progress: bool = True,
) -> RecoveryResult:
    """Recover the largest certified interpolation coefficient from Eq. (8a).

    The path is ``a(eta)=(1-eta)a_proj + eta*ahat``.  A deterministic grid is
    used in this first P0 implementation; it fails closed and is easy to audit.
    """

    if tau < 0 or not np.isfinite(tau):
        raise SetGRPOError("tau must be finite and non-negative")
    if grid_size < 2:
        raise SetGRPOError("grid_size must be at least 2")
    if margin_tolerance < 0 or not np.isfinite(margin_tolerance):
        raise SetGRPOError("margin_tolerance must be finite and non-negative")
    a_proj, _ = safe_anchor(ellipsoid, full_geometry)
    best: tuple[float, Array, Array, float, float] | None = None
    for eta in np.linspace(0.0, 1.0, grid_size):
        coefficients = (1.0 - eta) * a_proj + eta * ellipsoid.center
        full_update = full_geometry.update(coefficients)
        reward_update = reward_geometry.update(coefficients)
        # A zero update is not a training signal.  At threshold=0 it would
        # otherwise pass the non-strict full-margin check (0 >= 0).
        if float(np.linalg.norm(full_update)) <= 1e-12:
            continue
        full_margin = support_margin(full_update, ellipsoid, full_geometry, tau_num=tau_num)
        reward_margin = support_margin(reward_update, ellipsoid, reward_geometry, tau_num=tau_num_reward)
        if passes_margin_test(
            full_margin,
            reward_margin,
            threshold=tau,
            margin_tolerance=margin_tolerance,
            require_reward_progress=require_reward_progress,
        ):
            best = (float(eta), coefficients, full_update, full_margin, reward_margin)
    if best is None:
        return RecoveryResult(False, 0.0, None, None, None, None)
    eta, coefficients, update, full_margin, reward_margin = best
    return RecoveryResult(True, eta, coefficients, update, full_margin, reward_margin)


def passes_margin_test(
    full_margin: float,
    reward_margin: float,
    *,
    threshold: float,
    margin_tolerance: float,
    require_reward_progress: bool,
) -> bool:
    """Return whether certificate margins pass, preserving strict defaults.

    ``margin_tolerance`` is a diagnostic slack for numerical boundary audits.
    At the default ``0.0`` it preserves the original inequalities exactly:
    ``full_margin >= threshold`` and ``reward_margin > 0``.
    """

    if margin_tolerance <= 0.0:
        reward_ok = reward_margin > 0.0 if require_reward_progress else True
        return full_margin >= threshold and reward_ok
    reward_ok = reward_margin + margin_tolerance >= 0.0 if require_reward_progress else True
    return full_margin + margin_tolerance >= threshold and reward_ok


def clopper_pearson_upper_bound(failures: int, trials: int, *, confidence: float) -> float:
    """Return the one-sided Clopper-Pearson upper confidence bound."""

    if trials < 0 or failures < 0 or failures > trials:
        raise SetGRPOError("failures and trials must satisfy 0 <= failures <= trials")
    if not 0 < confidence < 1:
        raise SetGRPOError("confidence must be in (0, 1)")
    if trials == 0:
        return 1.0
    if failures == trials:
        return 1.0
    return float(beta_dist.ppf(confidence, failures + 1, trials - failures))


def select_risk_threshold(
    calibration_rows: Iterable[Mapping[str, object]],
    thresholds: Sequence[float],
    *,
    beta: float,
    gamma: float,
) -> SelectiveRiskResult:
    """Select ``tau_sel`` using Eq. (8b).

    ``calibration_rows`` must contain one row per risk-calibration microbatch
    with fields:

    - ``threshold``: candidate threshold value.
    - ``accepted``: whether Eq. (8a) accepted the batch for that threshold.
    - ``reversed``: whether the accepted update was reference-opposite.
    - optional ``yield``: tie-break quantity; defaults to accepted count.

    Rows for different thresholds may be materialized explicitly.
    """

    if not 0 < beta < 1:
        raise SetGRPOError("beta must be in (0, 1)")
    if not 0 < gamma < 1:
        raise SetGRPOError("gamma must be in (0, 1)")
    grid = tuple(float(t) for t in thresholds)
    if not grid:
        raise SetGRPOError("thresholds must be non-empty")
    if any((not np.isfinite(t) or t < 0) for t in grid):
        raise SetGRPOError("thresholds must be finite and non-negative")
    rows = tuple(calibration_rows)
    conf = 1.0 - gamma / len(grid)
    summaries: list[dict[str, float | int | bool]] = []
    for threshold in grid:
        materialized = [row for row in rows if float(row["threshold"]) == threshold]
        accepted = [row for row in materialized if bool(row.get("accepted", False))]
        failures = sum(1 for row in accepted if bool(row.get("reversed", False)))
        trials = len(accepted)
        upper = clopper_pearson_upper_bound(failures, trials, confidence=conf)
        yield_score = float(sum(float(row.get("yield", 1.0)) for row in accepted))
        selectable = trials > 0 and upper <= beta
        summaries.append(
            {
                "threshold": threshold,
                "trials": trials,
                "failures": failures,
                "upper_bound": upper,
                "yield": yield_score,
                "selectable": selectable,
            }
        )
    selectable_rows = [row for row in summaries if bool(row["selectable"])]
    if not selectable_rows:
        return SelectiveRiskResult(None, tuple(summaries))
    selected = max(selectable_rows, key=lambda row: (float(row["yield"]), -float(row["threshold"])))
    return SelectiveRiskResult(float(selected["threshold"]), tuple(summaries))


def _vector(value: Sequence[float] | Array, name: str) -> Array:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise SetGRPOError(f"{name} must be one-dimensional")
    if arr.size == 0:
        raise SetGRPOError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise SetGRPOError(f"{name} must contain only finite values")
    return arr


def _matrix(value: Sequence[Sequence[float]] | Array, name: str) -> Array:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise SetGRPOError(f"{name} must be two-dimensional")
    if 0 in arr.shape:
        raise SetGRPOError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise SetGRPOError(f"{name} must contain only finite values")
    return arr


def _assert_spd(matrix: Array, name: str) -> None:
    if matrix.shape[0] != matrix.shape[1]:
        raise SetGRPOError(f"{name} must be square")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise SetGRPOError(f"{name} must be positive definite") from exc


def _block_diag(blocks: Sequence[Array]) -> Array:
    if not blocks:
        raise SetGRPOError("at least one block is required")
    rows = sum(block.shape[0] for block in blocks)
    cols = sum(block.shape[1] for block in blocks)
    out = np.zeros((rows, cols), dtype=np.float64)
    r = c = 0
    for block in blocks:
        rr, cc = block.shape
        out[r : r + rr, c : c + cc] = block
        r += rr
        c += cc
    return out
