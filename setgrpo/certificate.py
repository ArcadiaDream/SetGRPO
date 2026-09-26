"""Frozen calibration rules and deploy-time coefficient certificates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .calibration import CoefficientSet, joint_coefficients, mahalanobis_radius, split_conformal_radius
from .data import rows_from_jsonl
from .geometry import LocalPolicyGeometry
from .recovery import GeometryGuidedSafeRecovery
from .risk import SelectiveRiskGatedOptimization


def freeze_rule(calibration_rows: Sequence[Mapping[str, Any]], risk_rows: Sequence[Mapping[str, Any]], *, alpha: float = 0.1, beta_target: float = 0.25, gamma: float = 0.25, thresholds: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)) -> dict[str, Any]:
    """Freeze the conformal radius and selective threshold."""

    residuals = [mahalanobis_radius(row["reference_coefficients"], row["estimated_coefficients"], row["shape"]) for row in calibration_rows]
    radius = split_conformal_radius(residuals, alpha=alpha)
    materialized = []
    for row in risk_rows:
        for threshold in thresholds:
            result = certify_row(row, radius=radius, threshold=float(threshold))
            materialized.append({"threshold": float(threshold), "accepted": result["accepted"], "reversed": result["reversed"], "yield": result["eta"]})
    selected = SelectiveRiskGatedOptimization(beta_target=beta_target, gamma=gamma).select(materialized, thresholds)
    if selected.selected_threshold is None:
        raise ValueError("risk calibration produced no admissible threshold")
    return {"alpha": alpha, "beta": beta_target, "gamma": gamma, "radius": radius, "selected_threshold": selected.selected_threshold, "calibration_rows": len(calibration_rows), "risk_rows": len(risk_rows)}


def certify_row(row: Mapping[str, Any], *, radius: float, threshold: float) -> dict[str, Any]:
    center = np.asarray(row["estimated_coefficients"], dtype=np.float64)
    coefficient_set = CoefficientSet(center, np.asarray(row["shape"], dtype=np.float64), radius)
    geometry = _geometry(row["geometry"])
    reward_geometry = _geometry(row.get("reward_geometry", row["geometry"]))
    recovery = GeometryGuidedSafeRecovery(coefficient_set, geometry, reward_geometry).recover(threshold=threshold)
    reversed_update = False
    if recovery.accepted and recovery.update is not None and row.get("reference_coefficients") is not None:
        reference_update = geometry.update(row["reference_coefficients"])
        reversed_update = bool(float(recovery.update @ reference_update) < 0.0)
    return {"accepted": recovery.accepted, "eta": recovery.eta, "coefficients": None if recovery.coefficients is None else recovery.coefficients.tolist(), "reversed": reversed_update}


class FrozenCertificateProvider:
    """Load frozen rows and emit coefficients for a current rollout group."""

    def __init__(self, certificate_path: str | Path, rule_path: str | Path, *, key: str = "microbatch_id", grid_size: int = 257) -> None:
        self.rows = {str(row[key]): row for row in rows_from_jsonl(str(certificate_path))}
        self.rule = json.loads(Path(rule_path).read_text(encoding="utf-8"))
        self.key = key
        self.grid_size = grid_size
        self.diagnostics: dict[str, dict[str, Any]] = {}

    def __call__(self, group_id: str, scores: Sequence[float], metadata: Mapping[str, Any] | None = None) -> Sequence[float] | None:
        row_id = str(metadata.get(self.key, group_id) if metadata else group_id)
        row = self.rows.get(row_id, self.rows.get("__default__"))
        if row is None:
            self.diagnostics[row_id] = {"status": "rejected", "reason": "missing_certificate"}
            return None
        row = dict(row)
        row["estimated_coefficients"] = joint_coefficients(scores).tolist()
        if metadata and metadata.get("setgrpo_jacobian") is not None:
            jacobian = np.asarray(metadata["setgrpo_jacobian"], dtype=np.float64)
            geometry = {"G": jacobian.tolist(), "P": np.eye(jacobian.shape[0]).tolist(), "h": np.zeros(jacobian.shape[0]).tolist()}
            row["geometry"] = geometry
            row["reward_geometry"] = geometry
        result = certify_row(row, radius=float(self.rule["radius"]), threshold=float(self.rule["selected_threshold"]))
        self.diagnostics[row_id] = {"status": "accepted" if result["accepted"] else "deferred", "eta": result["eta"], "reason": "accepted" if result["accepted"] else "certificate_constraint"}
        return result["coefficients"]


def _geometry(value: Mapping[str, Any]) -> LocalPolicyGeometry:
    return LocalPolicyGeometry(np.asarray(value["G"], dtype=np.float64), np.asarray(value["P"], dtype=np.float64), np.asarray(value["h"], dtype=np.float64))
