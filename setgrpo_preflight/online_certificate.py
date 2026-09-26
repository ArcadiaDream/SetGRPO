"""Online certificate coefficient provider using frozen geometry artifacts."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .core import CoefficientEllipsoid, LocalGeometry, SetGRPOError, group_standardized_coefficients, recover_progress, safe_anchor
from .microbatch import microbatch_row_from_scores
from .online import CoefficientProvider


class OnlineCertificateProvider:
    """Compute certified coefficients from current rewards and frozen rows."""

    def __init__(self, path: str | Path, *, rule_path: str | Path | None = None, key: str = "microbatch_id", grid_size: int = 257, variant: str = "full", discard_rate: float = 0.0) -> None:
        self.path = Path(path).expanduser()
        self.key = key
        self.grid_size = int(grid_size)
        self.variant = str(variant)
        self.discard_rate = float(discard_rate)
        self.last_diagnostics: dict[str, dict[str, float | str | bool]] = {}
        if not 0.0 <= self.discard_rate < 1.0:
            raise SetGRPOError("discard_rate must be in [0, 1)")
        self._rows = self._load()
        self._rule = json.loads(Path(rule_path).read_text(encoding="utf-8")) if rule_path else None
        self._norm_discard_cutoff = self._compute_norm_discard_cutoff() if self.variant == "norm_matched_discard" else None

    def _compute_norm_discard_cutoff(self) -> float | None:
        """Freeze the discard cutoff from calibration-row update norms.

        This makes norm-matched discard a distinct control: it removes the
        same fraction of the largest local-update norms, rather than hashing
        group ids as the matched-random control does.
        """
        norms = []
        for row in self._rows.values():
            try:
                g = np.asarray(row["geometry"]["G"], dtype=np.float64)
                norms.append(float(np.linalg.norm(g)))
            except (KeyError, TypeError, ValueError):
                continue
        if not norms or self.discard_rate <= 0.0:
            return None
        return float(np.quantile(np.asarray(norms), 1.0 - self.discard_rate))

    def _load(self) -> dict[str, Mapping[str, Any]]:
        if not self.path.is_file():
            raise FileNotFoundError(f"SetGRPO online certificate rows not found: {self.path}")
        rows: dict[str, Mapping[str, Any]] = {}
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, Mapping) or row.get(self.key) is None:
                    raise SetGRPOError(f"invalid certificate row {line_number}")
                rows[str(row[self.key])] = row
        return rows

    def __call__(self, group_id: str, scores: Sequence[float], metadata: Mapping[str, Any] | None) -> Sequence[float] | None:
        metadata_value = metadata.get(self.key) if metadata else None
        if metadata_value is None and metadata and isinstance(metadata.get("metadata"), Mapping):
            metadata_value = metadata["metadata"].get(self.key)
        row_id = str(metadata_value) if metadata_value is not None else str(group_id)
        exact_row = self._rows.get(row_id)
        row = exact_row or self._rows.get("__default__")
        using_default_shape_rule = exact_row is None and row is not None
        if row is None or row.get("accepted") is False:
            self.last_diagnostics[row_id] = {"status": "rejected", "reason": "missing_or_rejected_certificate"}
            return None
        stable_bucket = int.from_bytes(hashlib.sha256(row_id.encode("utf-8")).digest()[:8], "big") / float(2**64)
        if self.discard_rate and self.variant != "norm_matched_discard" and stable_bucket < self.discard_rate:
            self.last_diagnostics[row_id] = {"status": "deferred", "reason": "discard_rate"}
            return None
        try:
            current = np.asarray(scores, dtype=np.float64)
            if using_default_shape_rule:
                # Freeze the warmup construction protocol, not the first
                # warmup group's literal matrix.  Each future group receives
                # its own S_x from its current reward vector under the same
                # pre-frozen variance, ridge, shrinkage and condition cap.
                row_metadata = row.get("metadata", {})
                regularization = (
                    row_metadata.get("shape_regularization", {})
                    if isinstance(row_metadata, Mapping)
                    else {}
                )
                rebuilt = microbatch_row_from_scores(
                    microbatch_id=row_id,
                    estimated_scores=current.tolist(),
                    score_variance=(row_metadata.get("score_variance", 1.0) if isinstance(row_metadata, Mapping) else 1.0),
                    ridge=float(row_metadata.get("ridge", 1e-8)) if isinstance(row_metadata, Mapping) else 1e-8,
                    shape_shrinkage=float(regularization.get("shrinkage", 0.0)) if isinstance(regularization, Mapping) else 0.0,
                    shape_condition_cap=(regularization.get("condition_cap") if isinstance(regularization, Mapping) else None),
                    shape_preserve_trace=bool(regularization.get("preserve_trace", True)) if isinstance(regularization, Mapping) else True,
                )
                shape = np.asarray(rebuilt["shape"], dtype=np.float64)
            else:
                shape = np.asarray(row["shape"], dtype=np.float64)
            geometry_map = row["geometry"]
            reward_map = row.get("reward_geometry", geometry_map)
            dynamic_jacobian = metadata.get("setgrpo_jacobian") if metadata else None
            if dynamic_jacobian is not None and self.variant not in {"identity_geometry", "bonferroni_set", "spherical_set"}:
                dynamic = np.asarray(dynamic_jacobian, dtype=np.float64)
                if dynamic.ndim != 2:
                    raise SetGRPOError("dynamic Jacobian must be a two-dimensional matrix")
                # The actor RPC returns [projection_dim, rollout_n].  The
                # static P/h from score-space rows have rollout_n dimensions;
                # use the identity map in projected policy space instead.
                geometry_map = {
                    "G": dynamic.tolist(),
                    "P": np.eye(dynamic.shape[0], dtype=np.float64).tolist(),
                    "h": np.zeros(dynamic.shape[0], dtype=np.float64).tolist(),
                }
                reward_map = geometry_map
            if self.variant == "norm_matched_discard" and self._norm_discard_cutoff is not None:
                current_norm = float(np.linalg.norm(np.asarray(geometry_map["G"], dtype=np.float64)))
                if current_norm >= self._norm_discard_cutoff:
                    self.last_diagnostics[row_id] = {"status": "deferred", "reason": "norm_matched_discard"}
                    return None
            geometry = LocalGeometry(np.asarray(geometry_map["G"]), np.asarray(geometry_map["P"]), np.asarray(geometry_map["h"]))
            reward_geometry = LocalGeometry(np.asarray(reward_map["G"]), np.asarray(reward_map["P"]), np.asarray(reward_map["h"]))
            center = group_standardized_coefficients(current)
            if center.shape != shape.shape[:1]:
                raise SetGRPOError("frozen geometry dimension does not match rollout group")
            radius = float(self._rule["radius"] if self._rule else row["radius"])
            threshold = float(self._rule["selected_threshold"] if self._rule else row.get("threshold", row.get("selected_threshold", 0.0)))
            if self.variant == "identity_geometry":
                geometry = LocalGeometry(np.eye(center.size), np.eye(center.size), np.zeros(center.size))
                reward_geometry = geometry
            elif self.variant == "spherical_set":
                shape = np.eye(center.size) * max(float(np.trace(shape)) / center.size, 1e-8)
            elif self.variant == "bonferroni_set":
                shape = np.diag(np.maximum(np.diag(shape), 1e-8))
            elif self.variant == "reward_only_recovery":
                # Keep the score-space map and remove policy-geometry terms.
                geometry = reward_geometry
            elif self.variant == "direct_update_c":
                # Direct-Update-C calibrates the executed update itself: the
                # reward-side support constraint is disabled.
                reward_geometry = LocalGeometry(np.eye(center.size), np.eye(center.size), np.zeros(center.size))
            elif self.variant == "direct_align_c":
                # Direct-Align-C uses the direct update/reference alignment
                # rather than the decomposed full-plus-reward certificate.
                geometry = LocalGeometry(np.eye(center.size), np.eye(center.size), np.zeros(center.size))
            ellipsoid = CoefficientEllipsoid(center=center, shape=shape, radius=radius)
            if self.variant == "safe_anchor_only":
                anchor, update = safe_anchor(ellipsoid, geometry)
                return anchor.tolist() if np.linalg.norm(update) > 1e-12 else None
            if self.variant == "fixed_threshold":
                threshold = 0.0
            require_reward = self.variant not in {"policy_only_recovery", "direct_align_c"}
            result = recover_progress(
                ellipsoid,
                geometry,
                reward_geometry,
                tau=threshold,
                grid_size=self.grid_size,
                require_reward_progress=require_reward,
            )
            self.last_diagnostics[row_id] = {
                "status": "accepted" if result.accepted and result.coefficients is not None else "deferred",
                "reason": "accepted" if result.accepted and result.coefficients is not None else "certificate_constraint",
                "full_margin": float(result.full_margin) if result.full_margin is not None else float("nan"),
                "reward_margin": float(result.reward_margin) if result.reward_margin is not None else float("nan"),
                "eta_star": float(result.eta_star),
            }
            return None if not result.accepted or result.coefficients is None else result.coefficients.tolist()
        except KeyError as exc:
            raise SetGRPOError(f"certificate row {row_id} missing field {exc.args[0]!r}") from exc
