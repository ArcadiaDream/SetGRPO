"""Offline reward for SetGRPO LoRA smoke jobs.

This reward is intentionally *not* a rubric/grader reward.  Its only purpose is
to let a tiny GRPO LoRA job create a PEFT adapter artifact for policy-geometry
extraction without calling hosted APIs from compute nodes.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any


def _stable_unit_interval(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(digest[:8], "little") / float(2**64 - 1)


def _score_response(response: str, data_source: str = "") -> float:
    text = response or ""
    words = re.findall(r"\w+", text, flags=re.UNICODE)
    length_score = min(len(words) / 96.0, 1.0)
    nonempty = 1.0 if text.strip() else 0.0
    hash_jitter = _stable_unit_interval(f"{data_source}\n{text}")
    # Keep scores bounded and varied across sampled responses.  This is a
    # smoke-training signal only; downstream P0 certification does not use it.
    return max(0.0, min(1.0, 0.15 * nonempty + 0.60 * length_score + 0.25 * hash_jitter))


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str | None = None,
    extra_info: dict[str, Any] | None = None,
) -> float:
    return _score_response(solution_str, data_source)


def compute_score_batched(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[str] | None = None,
    extra_infos: list[dict[str, Any]] | None = None,
    **_: Any,
) -> list[dict[str, Any]]:
    if ground_truths is None:
        ground_truths = [None] * len(solution_strs)
    results = []
    for data_source, solution, ground_truth in zip(data_sources, solution_strs, ground_truths):
        score = _score_response(str(solution), str(data_source))
        results.append(
            {
                "score": score,
                "acc": score > 0.5,
                "ground_truth": ground_truth,
                "offline_lora_smoke_reward": True,
                "response_chars": len(str(solution or "")),
            }
        )
    return results
