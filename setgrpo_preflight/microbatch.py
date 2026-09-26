"""Build frozen SetGRPO microbatch rows from cross-scored transactions.

This is the missing bridge between the RubricSemVer P0 scorer and the offline
SetGRPO certificate code.  It turns old/new aggregate reward scores for one
frozen answer group into the JSON-friendly row schema consumed by
``setgrpo-radius``, ``setgrpo-risk``, and ``setgrpo-certify``.

The geometry here is intentionally conservative: a reward-vector proxy where
the local update direction is the standardized response-level coefficient
vector itself.  Full policy-gradient / LoRA Jacobian extraction is a later
training-system integration layer, not hidden in this offline adapter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from setgrpo_preflight.core import (
    SetGRPOError,
    block_diagonal_shape,
    group_standardization_jacobian,
    group_standardized_coefficients,
)


@dataclass(frozen=True)
class SetGRPOMicrobatchBuildSummary:
    """Summary emitted when materializing SetGRPO microbatch rows."""

    input_transactions: int
    built_rows: int
    skipped_rows: int
    estimated_record: str
    reference_record: str
    score_variance: float
    ridge: float
    shape_shrinkage: float
    shape_condition_cap: float | None
    shape_preserve_trace: bool
    split_filter: tuple[str, ...] | None
    skipped: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def microbatch_row_from_scores(
    *,
    microbatch_id: str,
    estimated_scores: Sequence[float],
    reference_scores: Sequence[float] | None = None,
    answer_ids: Sequence[str] | None = None,
    transaction_id: str | None = None,
    estimated_record: str = "new",
    reference_record: str = "old",
    score_variance: float | Sequence[float] | Sequence[Sequence[float]] = 1.0,
    ridge: float = 1e-8,
    shape_shrinkage: float = 0.0,
    shape_condition_cap: float | None = None,
    shape_preserve_trace: bool = True,
    radius: float | None = None,
) -> dict[str, Any]:
    """Create one SetGRPO certificate input row from response-level scores."""

    estimated = _score_vector(estimated_scores, "estimated_scores")
    reference = None if reference_scores is None else _score_vector(reference_scores, "reference_scores")
    if reference is not None and reference.shape != estimated.shape:
        raise SetGRPOError("reference_scores and estimated_scores must have the same length")
    if answer_ids is not None and len(answer_ids) != estimated.size:
        raise SetGRPOError("answer_ids length must match score length")

    estimated_coefficients = group_standardized_coefficients(estimated)
    reference_coefficients = None if reference is None else group_standardized_coefficients(reference)
    jacobian = group_standardization_jacobian(estimated)
    variance = _variance_matrix(score_variance, estimated.size)
    shape = block_diagonal_shape([jacobian], [variance], ridge=ridge)
    shape, shape_regularization = _regularize_shape(
        shape,
        shrinkage=shape_shrinkage,
        condition_cap=shape_condition_cap,
        preserve_trace=shape_preserve_trace,
    )

    identity = np.eye(estimated.size, dtype=np.float64)
    zero = np.zeros(estimated.size, dtype=np.float64)
    row: dict[str, Any] = {
        "microbatch_id": str(microbatch_id),
        "estimated_coefficients": estimated_coefficients.tolist(),
        "shape": shape.tolist(),
        "geometry": {
            "G": identity.tolist(),
            "P": identity.tolist(),
            "h": zero.tolist(),
        },
        "reward_geometry": {
            "G": identity.tolist(),
            "P": identity.tolist(),
            "h": zero.tolist(),
        },
        "metadata": {
            "transaction_id": transaction_id,
            "answer_ids": list(answer_ids) if answer_ids is not None else None,
            "estimated_record": estimated_record,
            "reference_record": reference_record if reference is not None else None,
            "estimated_scores": estimated.tolist(),
            "reference_scores": None if reference is None else reference.tolist(),
            "shape_method": "standardization_jacobian_score_covariance",
            "score_variance": _variance_metadata(score_variance),
            "ridge": ridge,
            "shape_regularization": shape_regularization,
            "geometry_method": "identity_reward_vector_proxy",
            "reward_geometry_valid": False,
        },
    }
    if reference_coefficients is not None:
        row["reference_coefficients"] = reference_coefficients.tolist()
    if radius is not None:
        row["radius"] = float(radius)
    return row


def microbatch_row_from_transaction(
    transaction_row: Mapping[str, Any],
    *,
    estimated_record: str = "new",
    reference_record: str = "old",
    score_variance: float | Sequence[float] | Sequence[Sequence[float]] = 1.0,
    ridge: float = 1e-8,
    shape_shrinkage: float = 0.0,
    shape_condition_cap: float | None = None,
    shape_preserve_trace: bool = True,
    radius: float | None = None,
) -> dict[str, Any]:
    """Create one SetGRPO row from a persisted ``VersionTransaction`` dict."""

    estimated_ids, estimated_scores = _aggregate_scores(transaction_row, estimated_record)
    if reference_record.lower() == "none":
        reference_scores = None
        reference_ids = estimated_ids
    else:
        reference_ids, reference_scores = _aggregate_scores(transaction_row, reference_record)
    if tuple(reference_ids) != tuple(estimated_ids):
        raise SetGRPOError("estimated and reference score records must use the same answer order")

    return microbatch_row_from_scores(
        microbatch_id=_microbatch_id(transaction_row),
        estimated_scores=estimated_scores,
        reference_scores=reference_scores,
        answer_ids=estimated_ids,
        transaction_id=str(transaction_row.get("transaction_id", "")) or None,
        estimated_record=estimated_record,
        reference_record=reference_record,
        score_variance=score_variance,
        ridge=ridge,
        shape_shrinkage=shape_shrinkage,
        shape_condition_cap=shape_condition_cap,
        shape_preserve_trace=shape_preserve_trace,
        radius=radius,
    )


def build_microbatch_rows_from_transactions(
    transaction_rows: Iterable[Mapping[str, Any]],
    *,
    estimated_record: str = "new",
    reference_record: str = "old",
    score_variance: float = 1.0,
    ridge: float = 1e-8,
    shape_shrinkage: float = 0.0,
    shape_condition_cap: float | None = None,
    shape_preserve_trace: bool = True,
    radius: float | None = None,
    split_filter: Iterable[str] | None = None,
    split_key: str = "split",
    skip_degenerate: bool = False,
) -> tuple[list[dict[str, Any]], SetGRPOMicrobatchBuildSummary]:
    """Materialize SetGRPO rows from many transaction rows."""

    rows = list(transaction_rows)
    allowed_splits = None if split_filter is None else tuple(str(item) for item in split_filter)
    output: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for row in rows:
        split = str(row.get(split_key, ""))
        if allowed_splits is not None and split not in allowed_splits:
            skipped.append(
                {
                    "transaction_id": str(row.get("transaction_id", "")),
                    "reason": f"split '{split}' not in filter",
                }
            )
            continue
        try:
            output.append(
                microbatch_row_from_transaction(
                    row,
                    estimated_record=estimated_record,
                    reference_record=reference_record,
                    score_variance=score_variance,
                    ridge=ridge,
                    shape_shrinkage=shape_shrinkage,
                    shape_condition_cap=shape_condition_cap,
                    shape_preserve_trace=shape_preserve_trace,
                    radius=radius,
                )
            )
        except Exception as exc:
            if not skip_degenerate:
                raise
            skipped.append(
                {
                    "transaction_id": str(row.get("transaction_id", "")),
                    "reason": str(exc),
                }
            )

    summary = SetGRPOMicrobatchBuildSummary(
        input_transactions=len(rows),
        built_rows=len(output),
        skipped_rows=len(skipped),
        estimated_record=estimated_record,
        reference_record=reference_record,
        score_variance=float(score_variance),
        ridge=float(ridge),
        shape_shrinkage=float(shape_shrinkage),
        shape_condition_cap=None if shape_condition_cap is None else float(shape_condition_cap),
        shape_preserve_trace=bool(shape_preserve_trace),
        split_filter=allowed_splits,
        skipped=tuple(skipped),
    )
    return output, summary


def _aggregate_scores(transaction_row: Mapping[str, Any], record: str) -> tuple[tuple[str, ...], np.ndarray]:
    record_key = _record_key(record)
    payload = transaction_row.get(record_key)
    if not isinstance(payload, Mapping):
        raise SetGRPOError(f"transaction row is missing {record_key}")
    answer_scores = payload.get("answer_scores")
    if not isinstance(answer_scores, Sequence) or isinstance(answer_scores, (str, bytes)):
        raise SetGRPOError(f"{record_key}.answer_scores must be a sequence")
    ids: list[str] = []
    scores: list[float] = []
    for item in answer_scores:
        if not isinstance(item, Mapping):
            raise SetGRPOError("answer score rows must be mappings")
        ids.append(str(item["answer_id"]))
        scores.append(float(item["aggregate_score"]))
    return tuple(ids), _score_vector(scores, f"{record_key}.aggregate_scores")


def _record_key(record: str) -> str:
    normalized = record.lower()
    if normalized in {"old", "old_scores"}:
        return "old_scores"
    if normalized in {"new", "new_scores"}:
        return "new_scores"
    raise SetGRPOError("record must be one of old, old_scores, new, new_scores, or none")


def _microbatch_id(transaction_row: Mapping[str, Any]) -> str:
    answer_group = transaction_row.get("answer_group")
    if isinstance(answer_group, Mapping) and answer_group.get("query_root"):
        return str(answer_group["query_root"])
    if transaction_row.get("query_root"):
        return str(transaction_row["query_root"])
    return str(transaction_row.get("transaction_id", "unknown"))


def _score_vector(scores: Sequence[float], name: str) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise SetGRPOError(f"{name} must be a one-dimensional sequence with at least two values")
    if not np.all(np.isfinite(values)):
        raise SetGRPOError(f"{name} must contain only finite values")
    return values


def _variance_matrix(
    score_variance: float | Sequence[float] | Sequence[Sequence[float]],
    size: int,
) -> np.ndarray:
    variance = np.asarray(score_variance, dtype=np.float64)
    if variance.ndim == 0:
        value = float(variance)
        if value < 0 or not np.isfinite(value):
            raise SetGRPOError("score_variance must be finite and non-negative")
        return value * np.eye(size, dtype=np.float64)
    if variance.ndim == 1:
        if variance.size != size:
            raise SetGRPOError("score_variance vector length must match score length")
        if np.any(variance < 0) or not np.all(np.isfinite(variance)):
            raise SetGRPOError("score_variance vector must be finite and non-negative")
        return np.diag(variance)
    if variance.ndim == 2:
        if variance.shape != (size, size):
            raise SetGRPOError("score_variance matrix must be square with score dimension")
        if not np.all(np.isfinite(variance)):
            raise SetGRPOError("score_variance matrix must be finite")
        return variance
    raise SetGRPOError("score_variance must be a scalar, vector, or matrix")


def _variance_metadata(score_variance: float | Sequence[float] | Sequence[Sequence[float]]) -> Any:
    value = np.asarray(score_variance, dtype=np.float64)
    if value.ndim == 0:
        return float(value)
    return value.tolist()


def _regularize_shape(
    shape: np.ndarray,
    *,
    shrinkage: float,
    condition_cap: float | None,
    preserve_trace: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Optionally regularize the coefficient-set shape before calibration.

    This is a construction-time diagnostic for P0: rows are rebuilt with the
    regularized ``S_x`` and the conformal radius must be recalibrated from the
    coverage split.  It is therefore different from post-hoc radius shrinking.
    """

    original = np.asarray(shape, dtype=np.float64)
    if original.ndim != 2 or original.shape[0] != original.shape[1]:
        raise SetGRPOError("shape must be square before regularization")
    if not np.all(np.isfinite(original)):
        raise SetGRPOError("shape must be finite before regularization")
    if shrinkage < 0.0 or shrinkage > 1.0 or not np.isfinite(shrinkage):
        raise SetGRPOError("shape_shrinkage must be finite in [0, 1]")
    if condition_cap is not None:
        if condition_cap <= 1.0 or not np.isfinite(condition_cap):
            raise SetGRPOError("shape_condition_cap must be finite and greater than 1")

    before_eigs = np.linalg.eigvalsh(original)
    if np.min(before_eigs) <= 0.0:
        raise SetGRPOError("shape must be positive definite before regularization")

    original_trace = float(np.trace(original))
    regularized = original.copy()
    enabled = bool(shrinkage > 0.0 or condition_cap is not None)

    if shrinkage > 0.0:
        dim = regularized.shape[0]
        target = (original_trace / dim) * np.eye(dim, dtype=np.float64)
        regularized = (1.0 - shrinkage) * regularized + shrinkage * target
        regularized = 0.5 * (regularized + regularized.T)

    if condition_cap is not None:
        eigvals, eigvecs = np.linalg.eigh(regularized)
        max_eig = float(np.max(eigvals))
        min_allowed = max_eig / float(condition_cap)
        eigvals = np.maximum(eigvals, min_allowed)
        regularized = (eigvecs * eigvals) @ eigvecs.T
        regularized = 0.5 * (regularized + regularized.T)

    if preserve_trace and enabled:
        regularized_trace = float(np.trace(regularized))
        if regularized_trace <= 0.0 or not np.isfinite(regularized_trace):
            raise SetGRPOError("regularized shape trace must be finite and positive")
        regularized *= original_trace / regularized_trace
        regularized = 0.5 * (regularized + regularized.T)

    after_eigs = np.linalg.eigvalsh(regularized)
    if np.min(after_eigs) <= 0.0:
        raise SetGRPOError("regularized shape must be positive definite")

    metadata = {
        "enabled": enabled,
        "shrinkage": float(shrinkage),
        "condition_cap": None if condition_cap is None else float(condition_cap),
        "preserve_trace": bool(preserve_trace),
        "original_trace": original_trace,
        "regularized_trace": float(np.trace(regularized)),
        "original_eigen_min": float(np.min(before_eigs)),
        "original_eigen_max": float(np.max(before_eigs)),
        "regularized_eigen_min": float(np.min(after_eigs)),
        "regularized_eigen_max": float(np.max(after_eigs)),
        "original_condition": float(np.max(before_eigs) / np.min(before_eigs)),
        "regularized_condition": float(np.max(after_eigs) / np.min(after_eigs)),
    }
    return regularized, metadata
