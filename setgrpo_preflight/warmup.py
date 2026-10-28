"""Build a frozen online SetGRPO rule from an in-process warmup pair."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .freeze_rules import freeze_rules
from .microbatch import microbatch_row_from_scores


def build_warmup_rows(
    groups: Iterable[Mapping[str, Any]],
    *,
    output_dir: str | Path,
    alpha: float = 0.1,
    beta: float = 0.25,
    gamma: float = 0.25,
    thresholds: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5),
    ridge: float = 1e-8,
    shape_shrinkage: float = 0.0,
    shape_condition_cap: float | None = 1000.0,
) -> dict[str, Any]:
    """Materialize paired reference/estimated warmup groups and freeze rules.

    Each group must contain ``reference_scores``, ``estimated_scores`` and a
    current-step projected Jacobian ``jacobian``.  Both score vectors are
    obtained from real grader calls; no reward proxy is used here.
    """

    rows: list[dict[str, Any]] = []
    for group in groups:
        reference = np.asarray(group["reference_scores"], dtype=np.float64)
        estimated = np.asarray(group["estimated_scores"], dtype=np.float64)
        jacobian = np.asarray(group["jacobian"], dtype=np.float64)
        if reference.ndim != 1 or estimated.shape != reference.shape:
            raise ValueError("warmup score vectors must be paired and one-dimensional")
        if jacobian.ndim != 2 or jacobian.shape[1] != estimated.size:
            raise ValueError("warmup Jacobian must have shape [projection_dim, rollout_n]")
        row = microbatch_row_from_scores(
            microbatch_id=str(group["microbatch_id"]),
            estimated_scores=estimated.tolist(),
            reference_scores=reference.tolist(),
            ridge=ridge,
            shape_shrinkage=shape_shrinkage,
            shape_condition_cap=shape_condition_cap,
        )
        geometry = {"G": jacobian.tolist(), "P": np.eye(jacobian.shape[0]).tolist(), "h": np.zeros(jacobian.shape[0]).tolist()}
        row["geometry"] = geometry
        row["reward_geometry"] = geometry
        row["metadata"]["geometry_method"] = "online_warmup_policy_jacobian"
        row["metadata"]["reward_geometry_valid"] = True
        rows.append(row)
    if not rows:
        raise ValueError("warmup produced no rows")
    # Future online groups have fresh ids; keep a validated shape template for
    # dynamic Jacobians while retaining the warmup rows for auditability.
    rows.append({**rows[0], "microbatch_id": "__default__"})

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    calibration_path = destination / "calibration_rows.jsonl"
    risk_path = destination / "risk_rows.jsonl"
    certificate_path = destination / "certificate_rows.jsonl"
    actual = rows[:-1]
    # Warmup is an in-process startup stage, but its calibration and risk
    # estimates must still use disjoint group objects.  With very small smoke
    # runs we fail clearly instead of silently reusing the same rows.
    if len(actual) < 3:
        raise ValueError("warmup requires at least three groups for disjoint calibration/risk splits")
    # Use disjoint, balanced splits.  The previous every-third split left
    # only about one third of warmup groups for risk calibration, which made
    # threshold selection unnecessarily brittle for small warmups.
    calibration_rows = actual[::2] or actual[:1]
    risk_rows = actual[1::2] or actual[1:]
    for path, persisted in ((calibration_path, calibration_rows), (risk_path, risk_rows), (certificate_path, rows)):
        with path.open("w", encoding="utf-8") as handle:
            for row in persisted:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    rule = freeze_rules(str(calibration_path), str(risk_path), str(destination / "frozen_rule.json"), alpha=alpha, beta=beta, gamma=gamma, thresholds=list(thresholds))
    return {"rows": len(rows) - 1, "calibration_path": str(calibration_path), "risk_path": str(risk_path), "certificate_path": str(certificate_path), "rule_path": str(destination / "frozen_rule.json"), "rule": rule}
