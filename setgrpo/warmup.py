"""Reference warmup protocol for paired reward calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import json
import numpy as np

from .data import coefficient_row, write_jsonl
from .certificate import freeze_rule


def build_frozen_artifacts(groups: Iterable[Mapping[str, Any]], output_dir: str | Path, *, alpha: float = 0.1, beta_target: float = 0.25, gamma: float = 0.25, thresholds: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)) -> dict[str, str]:
    """Build disjoint calibration/risk artifacts from paired warmup groups.

    Each group must contain ``microbatch_id``, ``estimated_scores``,
    ``reference_scores``, and a projected response Jacobian.
    """

    rows = []
    for group in groups:
        row = coefficient_row(group["microbatch_id"], group["estimated_scores"], reference_scores=group["reference_scores"])
        jacobian = np.asarray(group["jacobian"], dtype=np.float64)
        geometry = {"G": jacobian.tolist(), "P": np.eye(jacobian.shape[0]).tolist(), "h": np.zeros(jacobian.shape[0]).tolist()}
        row["geometry"] = geometry
        row["reward_geometry"] = geometry
        rows.append(row)
    if len(rows) < 3:
        raise ValueError("at least three warmup groups are required")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    calibration = rows[::2]
    risk = rows[1::2]
    default = dict(rows[0])
    default["microbatch_id"] = "__default__"
    write_jsonl(str(destination / "calibration_rows.jsonl"), calibration)
    write_jsonl(str(destination / "risk_rows.jsonl"), risk)
    write_jsonl(str(destination / "certificate_rows.jsonl"), [*rows, default])
    rule = freeze_rule(calibration, risk, alpha=alpha, beta_target=beta_target, gamma=gamma, thresholds=thresholds)
    (destination / "frozen_rule.json").write_text(json.dumps(rule, indent=2), encoding="utf-8")
    return {"calibration": str(destination / "calibration_rows.jsonl"), "risk": str(destination / "risk_rows.jsonl"), "certificates": str(destination / "certificate_rows.jsonl"), "rule": str(destination / "frozen_rule.json")}
