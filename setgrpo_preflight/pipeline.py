"""Offline SetGRPO P0 pipeline helpers.

The row schema here is intentionally explicit and JSON-friendly so calibration
artifacts can be produced before integrating with the online trainer.

Required execution row fields:

- ``estimated_coefficients``: point-estimate coefficient vector ``ahat_x``.
- ``shape``: positive-definite joint coefficient shape ``S_x``.
- ``radius``: frozen conformal radius ``q_alpha``.
- ``geometry``: mapping with ``G``, ``P``, and ``h`` for full update geometry.

Optional fields:

- ``reward_geometry``: same mapping for reward-only geometry.  If omitted, the
  full geometry with zero ``h`` is used.
- ``reference_coefficients``: reference vector used only for offline audit or
  risk calibration, never for deploy-time decisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from setgrpo_preflight.core import (
    CoefficientEllipsoid,
    LocalGeometry,
    RecoveryResult,
    SetGRPOError,
    mahalanobis_radius,
    recover_progress,
    select_risk_threshold,
    split_conformal_radius,
)


@dataclass(frozen=True)
class MicrobatchCertificate:
    """Deploy-time certificate plus optional offline audit outcome."""

    microbatch_id: str
    threshold: float
    accepted: bool
    eta_star: float
    coefficients: tuple[float, ...] | None
    full_margin: float | None
    reward_margin: float | None
    reversed_against_reference: bool | None
    update_norm: float | None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def conformal_radius_from_rows(
    calibration_rows: Iterable[Mapping[str, object]],
    *,
    alpha: float,
) -> float:
    """Compute ``q_alpha`` from rows containing estimated/reference coefficients."""

    residuals = [
        mahalanobis_radius(
            row["reference_coefficients"],
            row["estimated_coefficients"],
            row["shape"],
        )
        for row in calibration_rows
    ]
    return split_conformal_radius(residuals, alpha=alpha)


def certify_microbatch(
    row: Mapping[str, object],
    *,
    threshold: float,
    grid_size: int = 257,
    tau_num: float = 0.0,
    tau_num_reward: float = 0.0,
    margin_tolerance: float = 0.0,
) -> MicrobatchCertificate:
    """Run Eq. (8a)/(8c) for one row and optionally audit reversal."""

    microbatch_id = str(row.get("microbatch_id", row.get("query_root", "unknown")))
    try:
        ellipsoid = _ellipsoid_from_row(row)
        full_geometry = _geometry_from_mapping(row["geometry"])
        reward_geometry = _reward_geometry_from_row(row, full_geometry)
        metadata = row.get("metadata", {})
        proxy_reward_geometry = (
            isinstance(metadata, Mapping)
            and metadata.get("geometry_method") == "identity_reward_vector_proxy"
            and metadata.get("reward_geometry_valid") is not True
        )
        recovery = recover_progress(
            ellipsoid,
            full_geometry,
            reward_geometry,
            tau=threshold,
            tau_num=tau_num,
            tau_num_reward=tau_num_reward,
            grid_size=grid_size,
            margin_tolerance=margin_tolerance,
            require_reward_progress=not proxy_reward_geometry,
        )
    except SetGRPOError as exc:
        return MicrobatchCertificate(
            microbatch_id=microbatch_id,
            threshold=threshold,
            accepted=False,
            eta_star=0.0,
            coefficients=tuple(_zero_coefficients(row)),
            full_margin=None,
            reward_margin=None,
            reversed_against_reference=None,
            update_norm=None,
            error=str(exc),
        )
    reversed_against_reference = _audit_reversal(row, full_geometry, recovery)
    update_norm = None if recovery.update is None else float(np.linalg.norm(recovery.update))
    coefficients = (
        None
        if recovery.coefficients is None
        else tuple(float(value) for value in recovery.coefficients)
    )
    return MicrobatchCertificate(
        microbatch_id=microbatch_id,
        threshold=threshold,
        accepted=recovery.accepted,
        eta_star=recovery.eta_star,
        coefficients=coefficients,
        full_margin=recovery.full_margin,
        reward_margin=recovery.reward_margin,
        reversed_against_reference=reversed_against_reference,
        update_norm=update_norm,
    )


def _zero_coefficients(row: Mapping[str, object]) -> list[float]:
    values = row.get("estimated_coefficients", [])
    try:
        return [0.0 for _ in values]  # type: ignore[union-attr]
    except TypeError:
        return []


def materialize_risk_calibration_rows(
    rows: Iterable[Mapping[str, object]],
    thresholds: Sequence[float],
    *,
    grid_size: int = 257,
    tau_num: float = 0.0,
    tau_num_reward: float = 0.0,
    margin_tolerance: float = 0.0,
) -> list[dict[str, float | str | bool | None]]:
    """Evaluate all risk-calibration rows for every frozen threshold."""

    materialized: list[dict[str, float | str | bool | None]] = []
    for row in rows:
        for threshold in thresholds:
            cert = certify_microbatch(
                row,
                threshold=float(threshold),
                grid_size=grid_size,
                tau_num=tau_num,
                tau_num_reward=tau_num_reward,
                margin_tolerance=margin_tolerance,
            )
            materialized.append(
                {
                    **cert.to_dict(),
                    "reversed": bool(cert.reversed_against_reference) if cert.reversed_against_reference is not None else False,
                    "yield": cert.eta_star if cert.accepted else 0.0,
                }
            )
    return materialized


def select_threshold_from_microbatches(
    rows: Iterable[Mapping[str, object]],
    thresholds: Sequence[float],
    *,
    beta: float,
    gamma: float,
    grid_size: int = 257,
    margin_tolerance: float = 0.0,
) -> dict[str, object]:
    """Run risk calibration and select ``tau_sel`` from Eq. (8b)."""

    risk_rows = materialize_risk_calibration_rows(
        rows,
        thresholds,
        grid_size=grid_size,
        margin_tolerance=margin_tolerance,
    )
    selected = select_risk_threshold(risk_rows, thresholds, beta=beta, gamma=gamma)
    return {
        "selected_threshold": selected.selected_threshold,
        "threshold_rows": selected.threshold_rows,
        "risk_rows": risk_rows,
        "beta": beta,
        "gamma": gamma,
        "margin_tolerance": margin_tolerance,
    }


def _ellipsoid_from_row(row: Mapping[str, object]) -> CoefficientEllipsoid:
    return CoefficientEllipsoid(
        center=np.asarray(row["estimated_coefficients"], dtype=np.float64),
        shape=np.asarray(row["shape"], dtype=np.float64),
        radius=float(row["radius"]),
    )


def _geometry_from_mapping(mapping: object) -> LocalGeometry:
    if not isinstance(mapping, Mapping):
        raise SetGRPOError("geometry must be a mapping with G, P, and h")
    return LocalGeometry(
        G=np.asarray(mapping["G"], dtype=np.float64),
        P=np.asarray(mapping["P"], dtype=np.float64),
        h=np.asarray(mapping["h"], dtype=np.float64),
    )


def _reward_geometry_from_row(row: Mapping[str, object], full_geometry: LocalGeometry) -> LocalGeometry:
    if "reward_geometry" in row and row["reward_geometry"] is not None:
        return _geometry_from_mapping(row["reward_geometry"])
    return LocalGeometry(
        G=full_geometry.G,
        P=full_geometry.P,
        h=np.zeros_like(full_geometry.h),
    )


def _audit_reversal(
    row: Mapping[str, object],
    full_geometry: LocalGeometry,
    recovery: RecoveryResult,
) -> bool | None:
    if not recovery.accepted or recovery.update is None:
        return None
    if "reference_coefficients" not in row or row["reference_coefficients"] is None:
        return None
    reference_update = full_geometry.update(row["reference_coefficients"])
    return bool(float(np.dot(recovery.update, reference_update)) < 0.0)
