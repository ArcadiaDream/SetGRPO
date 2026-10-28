"""Command-line entry points for the public calibration workflow."""

from __future__ import annotations

import argparse
import json

from .certificate import freeze_rule
from .data import rows_from_jsonl


def calibrate() -> None:
    parser = argparse.ArgumentParser(description="Freeze a SetGRPO calibration and risk rule")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--risk", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5])
    args = parser.parse_args()
    rule = freeze_rule(rows_from_jsonl(args.calibration), rows_from_jsonl(args.risk), alpha=args.alpha, beta_target=args.beta, gamma=args.gamma, thresholds=args.thresholds)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(rule, handle, indent=2)
    print(json.dumps(rule, indent=2))


def certify() -> None:
    raise SystemExit("setgrpo-certify requires an application-specific rollout provider; use FrozenCertificateProvider in Python")
