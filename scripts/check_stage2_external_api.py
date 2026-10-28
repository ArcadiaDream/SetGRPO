#!/usr/bin/env python3
"""Minimal connectivity check for the stage-2 external judge API.

The script deliberately has no API-key argument. Set STAGE2_API_KEY (or the
existing environment-only OPENAI_API_KEY) before running it. It first checks
the endpoint's model list, then sends one one-token chat request pinned to the
explicitly configured model (gpt-5.6-luna by default). Any provider error is returned
with a non-zero exit code; no fallback model or endpoint is attempted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":  # pragma: no cover - CLI path mode
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.stage2_api import Stage2APIError, Stage2ExternalAPIClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds (default: 30).")
    parser.add_argument("--max-tokens", type=int, default=1, help="Smoke completion output budget (default: 1).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_tokens <= 0:
        print("ERROR: --max-tokens must be positive", file=sys.stderr)
        return 2
    try:
        client = Stage2ExternalAPIClient.from_env(timeout=args.timeout)
        model_ids = client.check_model()
        content = client.complete(
            [
                {"role": "system", "content": "Reply with OK only."},
                {"role": "user", "content": "Connectivity check."},
            ],
            temperature=0.0,
            top_p=1.0,
            max_tokens=args.max_tokens,
        )
    except (Stage2APIError, ValueError) as exc:
        status = f" status_code={exc.status_code}" if isinstance(exc, Stage2APIError) and exc.status_code else ""
        print(f"ERROR: stage-2 external API connectivity failed:{status} {exc}", file=sys.stderr)
        return 1
    print(
        "STAGE2_EXTERNAL_API_OK "
        f"endpoint={client.config.base_url} model={client.config.model} "
        f"advertised_models={','.join(model_ids)} response_chars={len(content.strip())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
