"""Certificate-backed coefficient provider for online SetGRPO training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .core import SetGRPOError


class CertificateCoefficientProvider:
    """Serve frozen certificate coefficients by rollout group identifier.

    The certificate file is loaded once on the trainer driver.  Each call
    returns the vector for one rollout group, or ``None`` when no accepted
    certificate exists for that group.
    """

    def __init__(self, path: str | Path, *, key: str = "microbatch_id") -> None:
        self.path = Path(path).expanduser()
        self.key = key
        self._coefficients = self._load()

    def _load(self) -> dict[str, tuple[float, ...]]:
        if not self.path.is_file():
            raise FileNotFoundError(f"SetGRPO certificate file not found: {self.path}")
        lookup: dict[str, tuple[float, ...]] = {}
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise SetGRPOError(f"certificate row {line_number} is not an object")
                row_id = row.get(self.key)
                coefficients = row.get("coefficients")
                if row_id is None or coefficients is None or row.get("accepted") is False:
                    continue
                values = np.asarray(coefficients, dtype=np.float64)
                if values.ndim != 1 or not np.all(np.isfinite(values)):
                    raise SetGRPOError(f"invalid coefficients in certificate row {line_number}")
                lookup[str(row_id)] = tuple(float(value) for value in values)
        return lookup

    def __call__(
        self,
        group_id: str,
        scores: Sequence[float],
        metadata: Mapping[str, Any] | None,
    ) -> Sequence[float] | None:
        lookup_id = str(group_id)
        if metadata is not None:
            candidate = metadata.get(self.key)
            if candidate is not None:
                lookup_id = str(candidate)
        values = self._coefficients.get(lookup_id)
        if values is None:
            return None
        if len(values) != len(scores):
            raise SetGRPOError(
                f"certificate length {len(values)} does not match rollout length {len(scores)} "
                f"for group {lookup_id}"
            )
        return values
