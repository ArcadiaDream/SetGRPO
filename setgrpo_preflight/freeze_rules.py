"""Create the immutable startup rule used by online SetGRPO training."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .pipeline import conformal_radius_from_rows, select_threshold_from_microbatches


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)

def freeze_rules(calibration_path: str, risk_path: str, output_path: str, *, alpha: float = .1, beta: float = .25, gamma: float = .25, thresholds: list[float] | None = None) -> dict[str, Any]:
    calibration = read_jsonl(calibration_path)
    risk = read_jsonl(risk_path)
    radius = conformal_radius_from_rows(calibration, alpha=alpha)
    rows = [{**row, "radius": radius} for row in risk]
    result = select_threshold_from_microbatches(rows, thresholds or [0.0, .05, .1, .2, .3, .5], beta=beta, gamma=gamma)
    selected = result.get("selected_threshold")
    if selected is None:
        summary = result.get("threshold_rows", [])
        raise RuntimeError(
            "risk calibration produced no selected threshold; "
            f"threshold_summary={summary}"
        )
    frozen = {"alpha": alpha, "beta": beta, "gamma": gamma, "radius": radius, "selected_threshold": float(selected), "calibration_rows": len(calibration), "risk_rows": len(risk)}
    write_json(output_path, frozen)
    return frozen


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--risk", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5])
    args = parser.parse_args()
    print(json.dumps(freeze_rules(args.calibration, args.risk, args.output, alpha=args.alpha, beta=args.beta, gamma=args.gamma, thresholds=args.thresholds), indent=2))
