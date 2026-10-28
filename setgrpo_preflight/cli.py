"""Standalone CLI for the SetGRPO preflight data and certificate pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import json


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

from .materialize import materialize_setgrpo_dataset
from .microbatch import build_microbatch_rows_from_transactions
from .pipeline import certify_microbatch, conformal_radius_from_rows, select_threshold_from_microbatches
from .splits import ROLES, assign_roles


def build_microbatches(args: argparse.Namespace) -> None:
    rows, summary = build_microbatch_rows_from_transactions(
        read_jsonl(args.input),
        estimated_record=args.estimated_record,
        reference_record=args.reference_record,
        score_variance=args.score_variance,
        ridge=args.ridge,
        shape_shrinkage=args.shape_shrinkage,
        shape_condition_cap=None if args.shape_condition_cap <= 0.0 else args.shape_condition_cap,
        shape_preserve_trace=not args.no_shape_preserve_trace,
        radius=args.radius,
        split_filter=args.split_filter,
        split_key=args.split_key,
        skip_degenerate=args.skip_degenerate,
    )
    output = Path(args.output)
    write_jsonl(output, rows, append=False)
    summary_path = Path(args.summary) if args.summary else output.with_suffix(".summary.json")
    write_json(summary_path, summary.to_dict())
    print(json.dumps(summary.to_dict(), indent=2), flush=True)


def freeze_radius(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    radius = conformal_radius_from_rows(rows, alpha=args.alpha)
    result = {"alpha": args.alpha, "radius": radius, "calibration_rows": len(rows), "input": args.input}
    write_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


def select_risk(args: argparse.Namespace) -> None:
    rows = _with_radius(read_jsonl(args.input), args.radius_json)
    result = select_threshold_from_microbatches(
        rows, args.thresholds, beta=args.beta, gamma=args.gamma, grid_size=args.grid_size
    )
    result["input_rows"] = len(rows)
    result["thresholds"] = args.thresholds
    write_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


def certify(args: argparse.Namespace) -> None:
    rows = _with_radius(read_jsonl(args.input), args.radius_json)
    certificates = [
        certify_microbatch(
            row,
            threshold=args.threshold,
            grid_size=args.grid_size,
            tau_num=args.tau_num,
            tau_num_reward=args.tau_num_reward,
        ).to_dict()
        for row in rows
    ]
    output = Path(args.output)
    write_jsonl(output, certificates, append=False)
    result = {
        "input_rows": len(rows),
        "accepted_rows": sum(bool(row["accepted"]) for row in certificates),
        "threshold": args.threshold,
        "output": str(output),
    }
    write_json(output.with_suffix(".summary.json"), result)
    print(json.dumps(result, indent=2), flush=True)


def materialize(args: argparse.Namespace) -> None:
    summary = materialize_setgrpo_dataset(
        input_path=args.input,
        certificates_path=args.certificates,
        output_path=args.output,
        input_key=args.input_key,
        certificate_key=args.certificate_key,
        output_key=args.output_key,
        coefficient_index=None if args.all_coefficients else args.coefficient_index,
        default_coefficient=args.default_coefficient,
        default_vector_length=args.default_vector_length,
        require_match=args.require_match,
        require_any_match=args.require_any_match,
        matched_only=args.matched_only,
        repeat_to_rows=args.repeat_to_rows,
        shuffle_seed=args.shuffle_seed,
        accepted_only=not args.include_rejected,
    )
    output = Path(args.output)
    summary_path = Path(args.summary) if args.summary else output.with_suffix(output.suffix + ".setgrpo_summary.json")
    write_json(summary_path, summary.to_dict())
    print(json.dumps(summary.to_dict(), indent=2), flush=True)


def _with_radius(rows: list[dict], radius_json: str) -> list[dict]:
    if not radius_json:
        return rows
    radius = float(json.loads(Path(radius_json).read_text(encoding="utf-8"))["radius"])
    return [{**row, "radius": radius} for row in rows]


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-microbatches")
    build.add_argument("--input", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--estimated-record", choices=("old", "new", "old_scores", "new_scores"), default="new")
    build.add_argument("--reference-record", choices=("old", "new", "old_scores", "new_scores", "none"), default="old")
    build.add_argument("--score-variance", type=float, default=1.0)
    build.add_argument("--ridge", type=float, default=1e-8)
    build.add_argument(
        "--shape-shrinkage",
        type=float,
        default=0.0,
        help="Convexly shrink S_x toward trace(S_x)/dim * I before conformal calibration.",
    )
    build.add_argument(
        "--shape-condition-cap",
        type=float,
        default=0.0,
        help="If positive, floor eigenvalues so the regularized S_x condition number is at most this cap.",
    )
    build.add_argument(
        "--no-shape-preserve-trace",
        action="store_true",
        help="Do not rescale the regularized shape back to the original trace.",
    )
    build.add_argument("--radius", type=float)
    build.add_argument("--split-filter", nargs="*")
    build.add_argument("--split-key", default="split")
    build.add_argument("--skip-degenerate", action="store_true")
    build.add_argument("--summary", default="")
    build.set_defaults(run=build_microbatches)

    radius = commands.add_parser("radius")
    radius.add_argument("--input", required=True)
    radius.add_argument("--output", required=True)
    radius.add_argument("--alpha", type=float, default=0.1)
    radius.set_defaults(run=freeze_radius)

    risk = commands.add_parser("risk")
    risk.add_argument("--input", required=True)
    risk.add_argument("--output", required=True)
    risk.add_argument("--thresholds", nargs="+", type=float, required=True)
    risk.add_argument("--beta", type=float, required=True)
    risk.add_argument("--gamma", type=float, required=True)
    risk.add_argument("--radius-json", default="")
    risk.add_argument("--grid-size", type=int, default=257)
    risk.set_defaults(run=select_risk)

    certificate = commands.add_parser("certify")
    certificate.add_argument("--input", required=True)
    certificate.add_argument("--output", required=True)
    certificate.add_argument("--threshold", type=float, required=True)
    certificate.add_argument("--radius-json", default="")
    certificate.add_argument("--grid-size", type=int, default=257)
    certificate.add_argument("--tau-num", type=float, default=0.0)
    certificate.add_argument("--tau-num-reward", type=float, default=0.0)
    certificate.set_defaults(run=certify)

    output = commands.add_parser("materialize")
    output.add_argument("--input", required=True)
    output.add_argument("--certificates", required=True)
    output.add_argument("--output", required=True)
    output.add_argument("--input-key", default="microbatch_id")
    output.add_argument("--certificate-key", default="microbatch_id")
    output.add_argument("--output-key", default="setgrpo_coefficients")
    output.add_argument("--coefficient-index", type=int, default=0)
    output.add_argument("--all-coefficients", action="store_true")
    output.add_argument("--default-coefficient", type=float, default=0.0)
    output.add_argument("--default-vector-length", type=int)
    output.add_argument("--require-match", action="store_true")
    output.add_argument("--require-any-match", action="store_true")
    output.add_argument("--matched-only", action="store_true")
    output.add_argument("--repeat-to-rows", type=int)
    output.add_argument("--shuffle-seed", type=int)
    output.add_argument("--include-rejected", action="store_true")
    output.add_argument("--summary", default="")
    output.set_defaults(run=materialize)

    splits = commands.add_parser("assign-splits")
    splits.add_argument("--input", required=True)
    splits.add_argument("--output", required=True)
    splits.add_argument("--summary", required=True)
    splits.add_argument("--salt", required=True)
    splits.add_argument("--proportions", nargs=len(ROLES), metavar="ROLE=VALUE", required=True)

    def assign_splits(args: argparse.Namespace) -> None:
        rows = read_jsonl(args.input)
        roots = []
        for row in rows:
            answer_group = row.get("answer_group")
            answer_group = answer_group if isinstance(answer_group, dict) else {}
            root = row.get("query_root") or row.get("prompt_id") or answer_group.get("query_root")
            if not root:
                raise ValueError("transaction row has no query_root/prompt_id")
            roots.append(str(root))
        proportions = dict(item.split("=", 1) for item in args.proportions)
        proportions = {role: float(proportions[role]) for role in ROLES}
        assignments = assign_roles(roots, proportions=proportions, salt=args.salt)
        output_rows = [{**row, "setgrpo_split": assignments[root], "setgrpo_split_salt": args.salt} for row, root in zip(rows, roots)]
        write_jsonl(args.output, output_rows, append=False)
        counts = {role: sum(value == role for value in assignments.values()) for role in ROLES}
        write_json(args.summary, {"input_rows": len(rows), "unique_roots": len(assignments), "salt": args.salt, "proportions": proportions, "counts": counts, "split_key": "setgrpo_split"})
        print(json.dumps({"input_rows": len(rows), "unique_roots": len(assignments), "counts": counts}, indent=2))

    splits.set_defaults(run=assign_splits)
    return root


def main() -> None:
    args = parser().parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
