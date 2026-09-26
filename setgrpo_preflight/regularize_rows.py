"""Regularize persisted SetGRPO row shapes without recomputing policy Jacobians."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .microbatch import _regularize_shape


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--condition-cap", required=True, type=float)
    parser.add_argument("--shrinkage", type=float, default=0.0)
    parser.add_argument("--no-preserve-trace", action="store_true")
    args = parser.parse_args()

    names = (
        "calibration_enriched.jsonl",
        "risk_calibration_enriched.jsonl",
        "train_enriched.jsonl",
        "evaluation_enriched.jsonl",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "condition_cap": args.condition_cap,
        "shrinkage": args.shrinkage,
        "preserve_trace": not args.no_preserve_trace,
        "files": {},
    }
    for name in names:
        source = args.input_dir / name
        destination = args.output_dir / name
        if not source.is_file():
            raise SystemExit(f"missing input: {source}")
        rows = []
        before = []
        after = []
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                shape, metadata = _regularize_shape(
                    np.asarray(row["shape"], dtype=np.float64),
                    shrinkage=args.shrinkage,
                    condition_cap=args.condition_cap,
                    preserve_trace=not args.no_preserve_trace,
                )
                row["shape"] = shape.tolist()
                row_metadata = dict(row.get("metadata", {}))
                row_metadata["shape_regularization"] = metadata
                row["metadata"] = row_metadata
                rows.append(row)
                before.append(float(metadata["original_condition"]))
                after.append(float(metadata["regularized_condition"]))
        with destination.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        summary["files"][name] = {
            "rows": len(rows),
            "max_condition_before": max(before),
            "max_condition_after": max(after),
        }
        print(f"{name}: rows={len(rows)} max_condition_after={max(after):.6g}", flush=True)
    (args.output_dir / "shape_regularization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
