"""Local policy geometry used by Geometry-Guided Safe Recovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class LocalPolicyGeometry:
    """Affine local update map ``u(a)=P(Ga+h)``."""

    response_jacobian: np.ndarray
    projection: np.ndarray
    offset: np.ndarray

    def __post_init__(self) -> None:
        jacobian = np.asarray(self.response_jacobian, dtype=np.float64)
        projection = np.asarray(self.projection, dtype=np.float64)
        offset = np.asarray(self.offset, dtype=np.float64)
        if jacobian.ndim != 2 or projection.ndim != 2 or offset.ndim != 1:
            raise ValueError("geometry fields have invalid ranks")
        if jacobian.shape[0] != offset.size or projection.shape != (offset.size, offset.size):
            raise ValueError("geometry dimensions are inconsistent")
        object.__setattr__(self, "response_jacobian", jacobian)
        object.__setattr__(self, "projection", projection)
        object.__setattr__(self, "offset", offset)

    @property
    def dimension(self) -> int:
        return self.response_jacobian.shape[1]

    def update(self, coefficients: Sequence[float]) -> np.ndarray:
        values = np.asarray(coefficients, dtype=np.float64)
        if values.shape != (self.dimension,):
            raise ValueError("coefficient dimension does not match response Jacobian")
        return self.projection @ (self.response_jacobian @ values + self.offset)

    def pullback(self, direction: Sequence[float]) -> np.ndarray:
        return self.response_jacobian.T @ (self.projection.T @ np.asarray(direction, dtype=np.float64))
