"""Adapters for feeding offline SetGRPO certificates into verl datasets.

The online trainer intentionally consumes only a tensor named
``setgrpo_coefficients``.  This module keeps the conversion from audit-time
certificate JSONL to train-time dataset rows explicit and reproducible.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class SetGRPOMergeSummary:
    """Summary of certificate-to-dataset materialization."""

    input_rows: int
    certificate_rows: int
    matched_rows: int
    defaulted_rows: int
    input_key: str
    certificate_key: str
    output_key: str
    coefficient_index: int | None
    all_coefficients: bool
    default_coefficient: float
    output_rows: int
    matched_only: bool
    repeat_to_rows: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_certificate_coefficients(
    certificate_rows: Iterable[Mapping[str, Any]],
    *,
    certificate_key: str = "microbatch_id",
    coefficient_index: int | None = 0,
    default_coefficient: float = 0.0,
    default_vector_length: int | None = None,
    accepted_only: bool = True,
) -> dict[str, float | list[float]]:
    """Build ``microbatch_id -> coefficient(s)`` from certificate rows.

    ``coefficients`` in the certificate is a vector.  ``coefficient_index``
    selects a scalar component.  Passing ``None`` keeps the full vector, which
    is the usual prompt-level dataset mode when rollout.n matches vector length.
    """

    rows = tuple(certificate_rows)
    vector_length = _resolve_vector_length(rows, default_vector_length) if coefficient_index is None else None
    lookup: dict[str, float | list[float]] = {}
    for row in rows:
        row_id = _row_id(row, certificate_key)
        if row_id is None:
            continue
        lookup[row_id] = _certificate_value(
            row,
            coefficient_index=coefficient_index,
            default_coefficient=default_coefficient,
            default_vector_length=vector_length,
            accepted_only=accepted_only,
        )
    return lookup


def merge_setgrpo_coefficients(
    dataset_rows: Sequence[Mapping[str, Any]],
    certificate_rows: Sequence[Mapping[str, Any]],
    *,
    input_key: str = "microbatch_id",
    certificate_key: str = "microbatch_id",
    output_key: str = "setgrpo_coefficients",
    coefficient_index: int | None = 0,
    default_coefficient: float = 0.0,
    default_vector_length: int | None = None,
    require_match: bool = False,
    require_any_match: bool = False,
    matched_only: bool = False,
    repeat_to_rows: int | None = None,
    shuffle_seed: int | None = None,
    accepted_only: bool = True,
) -> tuple[list[dict[str, Any]], SetGRPOMergeSummary]:
    """Return dataset rows with a ``setgrpo_coefficients`` column.

    ``input_key`` and ``certificate_key`` may be dotted paths, e.g.
    ``extra_info.index``.  Missing certificates fail closed to
    ``default_coefficient`` unless ``require_match`` is true.  When
    ``coefficient_index`` is ``None``, rows receive the full coefficient vector.
    """

    certificate_lookup = load_certificate_coefficients(
        certificate_rows,
        certificate_key=certificate_key,
        coefficient_index=coefficient_index,
        default_coefficient=default_coefficient,
        default_vector_length=default_vector_length,
        accepted_only=accepted_only,
    )
    fallback_coefficient = _default_coefficient_value(
        certificate_lookup.values(),
        coefficient_index=coefficient_index,
        default_coefficient=default_coefficient,
    )

    merged: list[dict[str, Any]] = []
    matched_rows = 0
    defaulted_rows = 0
    missing_ids: list[str] = []

    for row in dataset_rows:
        row_copy = dict(row)
        row_id = _row_id(row, input_key)
        matched = row_id is not None and row_id in certificate_lookup
        if matched:
            coefficient = certificate_lookup[row_id]
            matched_rows += 1
        else:
            coefficient = fallback_coefficient
            defaulted_rows += 1
            if row_id is not None:
                missing_ids.append(row_id)
            if matched_only:
                continue
        row_copy[output_key] = coefficient
        merged.append(row_copy)

    if require_match and missing_ids:
        preview = ", ".join(missing_ids[:5])
        suffix = "" if len(missing_ids) <= 5 else f", ... ({len(missing_ids)} total)"
        raise ValueError(f"missing SetGRPO certificates for dataset ids: {preview}{suffix}")
    if require_any_match and certificate_lookup and matched_rows == 0:
        raise ValueError(
            "no SetGRPO certificates matched dataset rows; check input_key/certificate_key "
            f"({input_key!r} vs {certificate_key!r}) before writing all-zero coefficients"
        )
    if repeat_to_rows is not None:
        if repeat_to_rows < 1:
            raise ValueError("repeat_to_rows must be positive")
        if merged and len(merged) < repeat_to_rows:
            source_rows = tuple(merged)
            merged = [dict(source_rows[index % len(source_rows)]) for index in range(repeat_to_rows)]
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(merged)

    summary = SetGRPOMergeSummary(
        input_rows=len(dataset_rows),
        certificate_rows=len(certificate_lookup),
        matched_rows=matched_rows,
        defaulted_rows=defaulted_rows,
        input_key=input_key,
        certificate_key=certificate_key,
        output_key=output_key,
        coefficient_index=coefficient_index,
        all_coefficients=coefficient_index is None,
        default_coefficient=float(default_coefficient),
        output_rows=len(merged),
        matched_only=bool(matched_only),
        repeat_to_rows=repeat_to_rows,
    )
    return merged, summary


def materialize_setgrpo_dataset(
    *,
    input_path: str | Path,
    certificates_path: str | Path,
    output_path: str | Path,
    input_key: str = "microbatch_id",
    certificate_key: str = "microbatch_id",
    output_key: str = "setgrpo_coefficients",
    coefficient_index: int | None = 0,
    default_coefficient: float = 0.0,
    default_vector_length: int | None = None,
    require_match: bool = False,
    require_any_match: bool = False,
    matched_only: bool = False,
    repeat_to_rows: int | None = None,
    shuffle_seed: int | None = None,
    accepted_only: bool = True,
) -> SetGRPOMergeSummary:
    """Read a JSONL/JSON/parquet dataset and write a SetGRPO-ready dataset."""

    dataset_rows = read_record_rows(input_path)
    certificate_rows = read_record_rows(certificates_path)
    merged, summary = merge_setgrpo_coefficients(
        dataset_rows,
        certificate_rows,
        input_key=input_key,
        certificate_key=certificate_key,
        output_key=output_key,
        coefficient_index=coefficient_index,
        default_coefficient=default_coefficient,
        default_vector_length=default_vector_length,
        require_match=require_match,
        require_any_match=require_any_match,
        matched_only=matched_only,
        repeat_to_rows=repeat_to_rows,
        shuffle_seed=shuffle_seed,
        accepted_only=accepted_only,
    )
    write_record_rows(output_path, merged)
    return summary


def read_record_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL, JSON list, or parquet rows into dictionaries."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(row) for row in payload]
        if isinstance(payload, Mapping) and isinstance(payload.get("rows"), list):
            return [dict(row) for row in payload["rows"]]
        raise ValueError(f"{path} must contain a JSON list or a mapping with a 'rows' list")
    if suffix in {".parquet", ".pq"}:
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"unsupported row file extension for {path}")


def write_record_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write rows as JSONL, JSON, or parquet based on the output suffix."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
        return
    if suffix == ".json":
        path.write_text(json.dumps(_jsonable(list(rows)), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    if suffix in {".parquet", ".pq"}:
        row_list = [dict(row) for row in rows]
        try:
            from datasets import Dataset

            Dataset.from_list(row_list).to_parquet(str(path))
        except ImportError:
            import pandas as pd

            pd.DataFrame(row_list).to_parquet(path, index=False)
        return
    raise ValueError(f"unsupported row file extension for {path}")


def _certificate_value(
    row: Mapping[str, Any],
    *,
    coefficient_index: int | None,
    default_coefficient: float,
    default_vector_length: int | None,
    accepted_only: bool,
) -> float | list[float]:
    coefficients = row.get("coefficients")
    if accepted_only and not bool(row.get("accepted", False)):
        return _zero_like_coefficients(
            coefficients, coefficient_index, default_coefficient, default_vector_length
        )
    if coefficients is None:
        return _zero_like_coefficients(
            coefficients, coefficient_index, default_coefficient, default_vector_length
        )
    if isinstance(coefficients, (int, float)):
        if coefficient_index not in (0, None):
            raise ValueError("scalar certificate coefficient only supports coefficient_index=0")
        return float(coefficients)
    if not isinstance(coefficients, Sequence) or isinstance(coefficients, (str, bytes)):
        raise ValueError("certificate coefficients must be a number or sequence of numbers")
    if coefficient_index is None:
        return [float(value) for value in coefficients]
    if coefficient_index < 0 or coefficient_index >= len(coefficients):
        raise ValueError(f"coefficient_index={coefficient_index} out of range for coefficients length {len(coefficients)}")
    return float(coefficients[coefficient_index])


def _zero_like_coefficients(
    coefficients: Any,
    coefficient_index: int | None,
    default_coefficient: float,
    default_vector_length: int | None,
) -> float | list[float]:
    if coefficient_index is not None:
        return float(default_coefficient)
    if isinstance(coefficients, Sequence) and not isinstance(coefficients, (str, bytes)):
        return [float(default_coefficient) for _ in coefficients]
    if default_vector_length is None:
        raise ValueError(
            "cannot infer SetGRPO coefficient vector length from null certificates; "
            "set default_vector_length"
        )
    return [float(default_coefficient) for _ in range(default_vector_length)]


def _resolve_vector_length(
    rows: Sequence[Mapping[str, Any]], explicit_length: int | None
) -> int | None:
    if explicit_length is not None and explicit_length < 1:
        raise ValueError("default_vector_length must be positive")
    lengths = {
        len(coefficients)
        for row in rows
        if isinstance((coefficients := row.get("coefficients")), Sequence)
        and not isinstance(coefficients, (str, bytes))
    }
    if len(lengths) > 1:
        raise ValueError(f"inconsistent SetGRPO coefficient vector lengths: {sorted(lengths)}")
    inferred = next(iter(lengths), None)
    if explicit_length is not None and inferred is not None and explicit_length != inferred:
        raise ValueError(
            f"default_vector_length={explicit_length} does not match certificate length {inferred}"
        )
    return explicit_length if explicit_length is not None else inferred


def _default_coefficient_value(
    values: Iterable[float | list[float]],
    *,
    coefficient_index: int | None,
    default_coefficient: float,
) -> float | list[float]:
    if coefficient_index is not None:
        return float(default_coefficient)
    for value in values:
        if isinstance(value, list):
            return [float(default_coefficient) for _ in value]
    return [float(default_coefficient)]


def _row_id(row: Mapping[str, Any], key_path: str) -> str | None:
    value = _resolve_key_path(row, key_path)
    if value is None:
        return None
    return str(value)


def _resolve_key_path(row: Mapping[str, Any], key_path: str) -> Any:
    value: Any = row
    for part in key_path.split("."):
        value = _decode_json_if_needed(value)
        if isinstance(value, Mapping):
            value = value.get(part)
        else:
            return None
    value = _decode_json_if_needed(value)
    return value


def _decode_json_if_needed(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    return value
