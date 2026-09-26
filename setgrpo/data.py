"""JSON-friendly certificate row helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .calibration import covariance_shape, joint_coefficients, standardization_jacobian


def coefficient_row(
    microbatch_id: str,
    estimated_scores: Sequence[float],
    *,
    reference_scores: Sequence[float] | None = None,
    score_variance: Sequence[Sequence[float]] | None = None,
    ridge: float = 1e-8,
) -> dict[str, Any]:
    """Create a portable calibration row without a model or grader dependency."""

    estimate = joint_coefficients(estimated_scores)
    jacobian = standardization_jacobian(estimated_scores)
    variance = np.eye(len(estimate)) if score_variance is None else np.asarray(score_variance, dtype=np.float64)
    shape = covariance_shape(jacobian, variance, ridge=ridge)
    identity = np.eye(len(estimate)).tolist()
    row: dict[str, Any] = {
        "microbatch_id": str(microbatch_id),
        "estimated_coefficients": estimate.tolist(),
        "shape": shape.tolist(),
        "geometry": {"G": identity, "P": identity, "h": np.zeros(len(estimate)).tolist()},
    }
    if reference_scores is not None:
        row["reference_coefficients"] = joint_coefficients(reference_scores).tolist()
    return row


def rows_from_jsonl(path: str) -> list[dict[str, Any]]:
    import json
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    import json
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
