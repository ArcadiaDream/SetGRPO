"""Independent deterministic SetGRPO protocol splits."""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping

ROLES = ("calibration", "risk_calibration", "train", "evaluation")

def assign_role(root: str, *, proportions: Mapping[str, float], salt: str) -> str:
    if not root:
        raise ValueError("root must be non-empty")
    if tuple(proportions) != ROLES:
        raise ValueError(f"proportions must define roles in order {ROLES}")
    if any(float(v) < 0 for v in proportions.values()) or abs(sum(map(float, proportions.values())) - 1.0) > 1e-9:
        raise ValueError("split proportions must be non-negative and sum to 1")
    draw = int.from_bytes(hashlib.sha256(f"{salt}\0{root}".encode()).digest()[:8], "big") / 2**64
    cursor = 0.0
    for role in ROLES:
        cursor += float(proportions[role])
        if draw < cursor:
            return role
    return ROLES[-1]

def assign_roles(roots: Iterable[str], *, proportions: Mapping[str, float], salt: str) -> dict[str, str]:
    return {root: assign_role(root, proportions=proportions, salt=salt) for root in dict.fromkeys(roots)}
