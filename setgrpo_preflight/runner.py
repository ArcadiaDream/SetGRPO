"""Reproducible SetGRPO preflight baselines, ablations, and audits.

The runner in this module operates on real frozen-answer transaction rows or
on their materialized microbatch rows.  It deliberately evaluates certificate
behavior before an expensive online policy run: conformal coverage, selective
risk, accepted yield, reversal rate, coefficient drift, and update margins.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from setgrpo_preflight.pipeline import (
    MicrobatchCertificate,
    certify_microbatch,
    conformal_radius_from_rows,
    select_threshold_from_microbatches,
)
from setgrpo_preflight.core import (
    CoefficientEllipsoid,
    LocalGeometry,
    SetGRPOError,
    direct_worst_case_support,
    group_standardization_jacobian,
    group_standardized_coefficients,
    linearized_standardization_error,
    mahalanobis_radius,
    passes_margin_test,
    safe_anchor,
    select_risk_threshold,
    split_conformal_radius,
    support_margin,
)


@dataclass(frozen=True)
class PreflightVariant:
    """One baseline or SetGRPO component ablation."""

    name: str
    description: str
    use_conformal: bool = True
    use_recovery: bool = True
    use_selective_risk: bool = True
    require_reward_progress: bool = True
    grpo_baseline: bool = False
    p0e_baseline: str | None = None


PREFLIGHT_VARIANTS: dict[str, PreflightVariant] = {
    "grpo": PreflightVariant(
        name="grpo",
        description="Vanilla group-standardized coefficients without conformal filtering.",
        use_conformal=False,
        use_recovery=False,
        use_selective_risk=False,
        require_reward_progress=False,
        grpo_baseline=True,
    ),
    "setgrpo_full": PreflightVariant(
        name="setgrpo_full",
        description="Full offline SetGRPO certificate and recovery pipeline.",
    ),
    "no_conformal": PreflightVariant(
        name="no_conformal",
        description="Ablation with zero conformal radius.",
        use_conformal=False,
    ),
    "no_recovery": PreflightVariant(
        name="no_recovery",
        description="Ablation that only tests the point estimate and skips anchor recovery.",
        use_recovery=False,
    ),
    "no_selective_risk": PreflightVariant(
        name="no_selective_risk",
        description="Ablation that fixes the selective threshold to zero.",
        use_selective_risk=False,
    ),
    "no_reward_constraint": PreflightVariant(
        name="no_reward_constraint",
        description="Ablation that certifies full-update progress without reward-only progress.",
        require_reward_progress=False,
    ),
    "safe_anchor_only": PreflightVariant(
        name="safe_anchor_only",
        description="Ablation that executes only the robust safe-anchor projection.",
        p0e_baseline="safe_anchor_only",
    ),
    "spherical": PreflightVariant(
        name="spherical",
        description="P0-E baseline with a spherical coefficient set instead of Jacobian-shaped ellipsoids.",
        p0e_baseline="spherical",
    ),
    "bonferroni": PreflightVariant(
        name="bonferroni",
        description="P0-E baseline using Bonferroni coordinate intervals in coefficient space.",
        p0e_baseline="bonferroni",
    ),
    "direct_update_c": PreflightVariant(
        name="direct_update_c",
        description="P0-E direct conformal baseline in local update space.",
        p0e_baseline="direct_update_c",
    ),
    "direct_align_c": PreflightVariant(
        name="direct_align_c",
        description="P0-E direct conformal baseline on nominal/reference update alignment.",
        p0e_baseline="direct_align_c",
    ),
    "matched_random_dropout": PreflightVariant(
        name="matched_random_dropout",
        description="P0-E matched random rejection baseline with deterministic prompt-level sampling.",
        p0e_baseline="matched_random_dropout",
    ),
    "anchor_to_nominal_recovery_audit": PreflightVariant(
        name="anchor_to_nominal_recovery_audit",
        description=(
            "PDF-aligned audit of the safe-anchor to nominal GRPO recovery path; "
            "uses the same certificate rule as full SetGRPO and reports path progress diagnostics."
        ),
        p0e_baseline="anchor_to_nominal_recovery_audit",
    ),
    "support_margin_decomposition_audit": PreflightVariant(
        name="support_margin_decomposition_audit",
        description=(
            "Audit-only decomposition of support margins into nominal support, "
            "pulled-back uncertainty penalty, radius, and final margin."
        ),
        p0e_baseline="support_margin_decomposition_audit",
    ),
    "radius_scale_sweep_audit": PreflightVariant(
        name="radius_scale_sweep_audit",
        description=(
            "Audit-only sweep over scaled conformal radii to identify when "
            "support penalties stop overwhelming nominal update progress."
        ),
        p0e_baseline="radius_scale_sweep_audit",
    ),
    "identity_geometry": PreflightVariant(
        name="identity_geometry",
        description="Identity geometry ablation using the audited no-conformal path.",
        use_conformal=False,
    ),
    "reward_only_recovery": PreflightVariant(
        name="reward_only_recovery",
        description="Reward-margin-only recovery ablation.",
        p0e_baseline="safe_anchor_only",
    ),
    "policy_only_recovery": PreflightVariant(
        name="policy_only_recovery",
        description="Policy-geometry-only recovery ablation.",
        require_reward_progress=False,
    ),
    "fixed_threshold": PreflightVariant(
        name="fixed_threshold",
        description="Fixed-threshold selective-risk ablation.",
        use_selective_risk=False,
    ),
    "norm_matched_discard": PreflightVariant(
        name="norm_matched_discard",
        description="Update-norm-matched discard control.",
        p0e_baseline="matched_random_dropout",
    ),
}

# Canonical paper ablations.  ``setgrpo_full`` is the reference arm and is
# intentionally excluded from this list; GRPO is a main-table baseline.
CANONICAL_ABLATIONS = (
    "identity_geometry", "bonferroni_set", "spherical_set", "safe_anchor_only",
    "reward_only_recovery", "policy_only_recovery", "fixed_threshold",
    "direct_update_c", "direct_align_c", "matched_random_discard", "norm_matched_discard",
)

# Keep the paper names as first-class variants rather than silently mapping
# them to the older audit names.
PREFLIGHT_VARIANTS["bonferroni_set"] = PreflightVariant(
    name="bonferroni_set", description="Coordinate-wise Bonferroni coefficient set.", p0e_baseline="bonferroni"
)
PREFLIGHT_VARIANTS["spherical_set"] = PreflightVariant(
    name="spherical_set", description="Spherical coefficient set with matched trace.", p0e_baseline="spherical"
)

RUN_DIRECTORY_VARIANTS = {
    "run_1": "grpo",
    "run_2": "setgrpo_full",
    "run_3": "no_conformal",
    "run_4": "no_recovery",
    "run_5": "no_selective_risk",
    "run_6": "no_reward_constraint",
    "run_7": "safe_anchor_only",
    "run_8": "spherical",
    "run_9": "bonferroni",
    "run_10": "direct_update_c",
    "run_11": "direct_align_c",
    "run_12": "matched_random_dropout",
    "run_13": "anchor_to_nominal_recovery_audit",
    "run_14": "support_margin_decomposition_audit",
    "run_15": "radius_scale_sweep_audit",
}


def variant_for_output_directory(name: str) -> str:
    """Return the fixed experiment variant associated with a run directory."""

    normalized = name.rstrip("/\\").split("/")[-1].split("\\")[-1]
    if normalized not in RUN_DIRECTORY_VARIANTS:
        choices = ", ".join(sorted(RUN_DIRECTORY_VARIANTS))
        raise ValueError(f"out_dir must end in one of {choices}, or pass --variant explicitly")
    return RUN_DIRECTORY_VARIANTS[normalized]


def run_preflight_experiment(
    coverage_rows: Sequence[Mapping[str, Any]],
    risk_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    *,
    variant_name: str,
    alpha: float = 0.1,
    beta: float = 0.05,
    gamma: float = 0.05,
    thresholds: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0),
    grid_size: int = 257,
    matched_acceptance_rate: float | None = None,
    seed: int = 2026,
    margin_tolerance: float = 0.0,
    radius_scales: Sequence[float] = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01),
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute one preflight baseline/ablation on frozen microbatch rows.

    Returns ``(final_info, certificates, coverage_audit)``.  Reference
    coefficients are used only for offline evaluation metrics and never for
    certificate decisions.
    """

    if variant_name not in PREFLIGHT_VARIANTS:
        raise ValueError(f"unknown variant {variant_name!r}; choose from {sorted(PREFLIGHT_VARIANTS)}")
    if not coverage_rows or not risk_rows or not evaluation_rows:
        raise ValueError("coverage, risk, and evaluation splits must all be non-empty")
    if margin_tolerance < 0.0 or not np.isfinite(margin_tolerance):
        raise ValueError("margin_tolerance must be finite and non-negative")
    radius_scales = tuple(float(value) for value in radius_scales)
    if not radius_scales or any((not np.isfinite(value) or value <= 0.0) for value in radius_scales):
        raise ValueError("radius_scales must be finite positive values")
    variant = PREFLIGHT_VARIANTS[variant_name]
    p0a_diagnostics = _p0a_diagnostics([*coverage_rows, *risk_rows, *evaluation_rows])

    if variant.p0e_baseline == "spherical":
        return _run_spherical_experiment(
            coverage_rows,
            risk_rows,
            evaluation_rows,
            variant=variant,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            thresholds=thresholds,
            grid_size=grid_size,
            margin_tolerance=margin_tolerance,
            p0a_diagnostics=p0a_diagnostics,
        )
    if variant.p0e_baseline == "bonferroni":
        return _run_bonferroni_experiment(
            coverage_rows,
            risk_rows,
            evaluation_rows,
            variant=variant,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            thresholds=thresholds,
            margin_tolerance=margin_tolerance,
            p0a_diagnostics=p0a_diagnostics,
        )
    if variant.p0e_baseline in {"direct_update_c", "direct_align_c"}:
        return _run_direct_conformal_experiment(
            coverage_rows,
            risk_rows,
            evaluation_rows,
            variant=variant,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            thresholds=thresholds,
            margin_tolerance=margin_tolerance,
            p0a_diagnostics=p0a_diagnostics,
        )
    if variant.p0e_baseline == "matched_random_dropout":
        return _run_matched_random_dropout(
            coverage_rows,
            risk_rows,
            evaluation_rows,
            variant=variant,
            alpha=alpha,
            matched_acceptance_rate=matched_acceptance_rate,
            seed=seed,
            margin_tolerance=margin_tolerance,
            p0a_diagnostics=p0a_diagnostics,
        )
    if variant.p0e_baseline == "anchor_to_nominal_recovery_audit":
        return _run_anchor_to_nominal_recovery_audit(
            coverage_rows,
            risk_rows,
            evaluation_rows,
            variant=variant,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            thresholds=thresholds,
            grid_size=grid_size,
            margin_tolerance=margin_tolerance,
            p0a_diagnostics=p0a_diagnostics,
        )
    if variant.p0e_baseline == "support_margin_decomposition_audit":
        return _run_support_margin_decomposition_audit(
            coverage_rows,
            risk_rows,
            evaluation_rows,
            variant=variant,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            thresholds=thresholds,
            grid_size=grid_size,
            margin_tolerance=margin_tolerance,
            p0a_diagnostics=p0a_diagnostics,
        )
    if variant.p0e_baseline == "radius_scale_sweep_audit":
        return _run_radius_scale_sweep_audit(
            coverage_rows,
            risk_rows,
            evaluation_rows,
            variant=variant,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            thresholds=thresholds,
            grid_size=grid_size,
            margin_tolerance=margin_tolerance,
            radius_scales=radius_scales,
            p0a_diagnostics=p0a_diagnostics,
        )

    calibrated_radius = conformal_radius_from_rows(coverage_rows, alpha=alpha)
    deployed_radius = calibrated_radius if variant.use_conformal else 0.0
    coverage_audit = _coverage_audit(coverage_rows, deployed_radius)

    prepared_risk = [_with_radius(row, deployed_radius) for row in risk_rows]
    prepared_evaluation = [_with_radius(row, deployed_radius) for row in evaluation_rows]

    threshold_summary: dict[str, Any] | None = None
    fail_closed_reason: str | None = None
    if variant.use_conformal and not np.isfinite(deployed_radius):
        selected_threshold = None
        fail_closed_reason = "split_conformal_radius_is_infinite; calibration split is too small for requested alpha"
    elif variant.grpo_baseline or not variant.use_selective_risk:
        selected_threshold = 0.0
    else:
        threshold_summary = select_threshold_from_microbatches(
            prepared_risk,
            thresholds,
            beta=beta,
            gamma=gamma,
            grid_size=grid_size,
            margin_tolerance=margin_tolerance,
        )
        selected = threshold_summary.get("selected_threshold")
        selected_threshold = None if selected is None else float(selected)

    if selected_threshold is None:
        certificates = [_rejected_certificate(row, threshold=0.0) for row in prepared_evaluation]
    elif variant.grpo_baseline:
        certificates = [_baseline_certificate(row) for row in prepared_evaluation]
    elif variant.p0e_baseline == "safe_anchor_only":
        certificates = [
            _safe_anchor_only_certificate(
                row,
                threshold=selected_threshold,
                margin_tolerance=margin_tolerance,
            )
            for row in prepared_evaluation
        ]
    elif variant.use_recovery and variant.require_reward_progress:
        certificates = [
            certify_microbatch(
                row,
                threshold=selected_threshold,
                grid_size=grid_size,
                margin_tolerance=margin_tolerance,
            ).to_dict()
            for row in prepared_evaluation
        ]
    else:
        certificates = [
            _certify_variant(
                row,
                threshold=selected_threshold,
                grid_size=grid_size,
                use_recovery=variant.use_recovery,
                require_reward_progress=variant.require_reward_progress,
                margin_tolerance=margin_tolerance,
            )
            for row in prepared_evaluation
        ]

    metrics = _summarize_certificates(certificates, prepared_evaluation)
    empirical_coverage = float(np.mean([row["covered"] for row in coverage_audit]))
    metrics.update(
        {
            "calibrated_radius": float(calibrated_radius),
            "deployed_radius": float(deployed_radius),
            "coverage": empirical_coverage,
            "coverage_target": 1.0 - alpha,
            "coverage_gap": empirical_coverage - (1.0 - alpha),
            "selected_threshold": selected_threshold,
            "fail_closed_reason": fail_closed_reason,
        }
    )
    final_info = {
        "variant": asdict(variant),
        "config": {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "thresholds": [float(value) for value in thresholds],
            "grid_size": grid_size,
            "margin_tolerance": margin_tolerance,
            "radius_scales": [float(value) for value in radius_scales],
        },
        "split_sizes": {
            "coverage": len(coverage_rows),
            "risk": len(risk_rows),
            "evaluation": len(evaluation_rows),
        },
        "metrics": metrics,
        "risk_selection": threshold_summary,
        "p0_diagnostics": {
            "p0_a": p0a_diagnostics,
            "p0_gate_report": _p0_gate_report(metrics, p0a_diagnostics, variant_name=variant_name),
        },
    }
    return final_info, certificates, coverage_audit


def _coverage_audit(rows: Sequence[Mapping[str, Any]], radius: float) -> list[dict[str, Any]]:
    audit = []
    for row in rows:
        residual = mahalanobis_radius(
            row["reference_coefficients"], row["estimated_coefficients"], row["shape"]
        )
        audit.append(
            {
                "microbatch_id": str(row.get("microbatch_id", "unknown")),
                "residual_radius": residual,
                "deployed_radius": radius,
                "covered": bool(residual <= radius),
            }
        )
    return audit


def _run_spherical_experiment(
    coverage_rows: Sequence[Mapping[str, Any]],
    risk_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    *,
    variant: PreflightVariant,
    alpha: float,
    beta: float,
    gamma: float,
    thresholds: Sequence[float],
    grid_size: int,
    margin_tolerance: float,
    p0a_diagnostics: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    radius = split_conformal_radius(
        [_coefficient_l2_residual(row) for row in coverage_rows],
        alpha=alpha,
    )
    coverage_audit = [
        {
            "microbatch_id": str(row.get("microbatch_id", "unknown")),
            "residual_radius": _coefficient_l2_residual(row),
            "deployed_radius": radius,
            "covered": bool(_coefficient_l2_residual(row) <= radius),
        }
        for row in coverage_rows
    ]
    prepared_risk = [_with_spherical_shape(row, radius) for row in risk_rows]
    prepared_evaluation = [_with_spherical_shape(row, radius) for row in evaluation_rows]
    if not np.isfinite(radius):
        threshold_summary = None
        selected_threshold = None
        certificates = [_rejected_certificate(row, threshold=0.0) for row in prepared_evaluation]
        fail_closed_reason = "spherical_split_conformal_radius_is_infinite"
    else:
        threshold_summary = select_threshold_from_microbatches(
            prepared_risk,
            thresholds,
            beta=beta,
            gamma=gamma,
            grid_size=grid_size,
            margin_tolerance=margin_tolerance,
        )
        selected = threshold_summary.get("selected_threshold")
        selected_threshold = None if selected is None else float(selected)
        certificates = (
            [_rejected_certificate(row, threshold=0.0) for row in prepared_evaluation]
            if selected_threshold is None
            else [
                certify_microbatch(
                    row,
                    threshold=selected_threshold,
                    grid_size=grid_size,
                    margin_tolerance=margin_tolerance,
                ).to_dict()
                for row in prepared_evaluation
            ]
        )
        fail_closed_reason = None
    return _finalize_variant(
        variant=variant,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        thresholds=thresholds,
        grid_size=grid_size,
        coverage_rows=coverage_rows,
        risk_rows=risk_rows,
        evaluation_rows=evaluation_rows,
        certificates=certificates,
        coverage_audit=coverage_audit,
        selected_threshold=selected_threshold,
        calibrated_radius=radius,
        deployed_radius=radius,
        threshold_summary=threshold_summary,
        p0a_diagnostics=p0a_diagnostics,
        margin_tolerance=margin_tolerance,
        extra_metrics={"fail_closed_reason": fail_closed_reason},
    )


def _run_bonferroni_experiment(
    coverage_rows: Sequence[Mapping[str, Any]],
    risk_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    *,
    variant: PreflightVariant,
    alpha: float,
    beta: float,
    gamma: float,
    thresholds: Sequence[float],
    margin_tolerance: float,
    p0a_diagnostics: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    radii = _bonferroni_radii(coverage_rows, alpha=alpha)
    coverage_audit = [
        {
            "microbatch_id": str(row.get("microbatch_id", "unknown")),
            "residual_radius": float(np.max(np.abs(_coefficient_delta(row)) / np.maximum(radii, 1e-12))),
            "deployed_radius": 1.0,
            "covered": bool(np.all(np.abs(_coefficient_delta(row)) <= radii)),
        }
        for row in coverage_rows
    ]
    risk_materialized = _materialize_static_certificates(
        risk_rows,
        thresholds,
        lambda row, threshold: _bonferroni_certificate(
            row,
            radii=radii,
            threshold=threshold,
            margin_tolerance=margin_tolerance,
        ),
    )
    selected = select_risk_threshold(risk_materialized, thresholds, beta=beta, gamma=gamma)
    selected_threshold = selected.selected_threshold
    certificates = (
        [_rejected_certificate(row, threshold=0.0) for row in evaluation_rows]
        if selected_threshold is None
        else [
            _bonferroni_certificate(
                row,
                radii=radii,
                threshold=selected_threshold,
                margin_tolerance=margin_tolerance,
            )
            for row in evaluation_rows
        ]
    )
    return _finalize_variant(
        variant=variant,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        thresholds=thresholds,
        grid_size=1,
        coverage_rows=coverage_rows,
        risk_rows=risk_rows,
        evaluation_rows=evaluation_rows,
        certificates=certificates,
        coverage_audit=coverage_audit,
        selected_threshold=selected_threshold,
        calibrated_radius=float(np.max(radii)),
        deployed_radius=float(np.max(radii)),
        threshold_summary={
            "selected_threshold": selected.selected_threshold,
            "threshold_rows": selected.threshold_rows,
            "risk_rows": risk_materialized,
            "beta": beta,
            "gamma": gamma,
            "margin_tolerance": margin_tolerance,
        },
        p0a_diagnostics=p0a_diagnostics,
        margin_tolerance=margin_tolerance,
        extra_metrics={"bonferroni_radii": radii.tolist()},
    )


def _run_direct_conformal_experiment(
    coverage_rows: Sequence[Mapping[str, Any]],
    risk_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    *,
    variant: PreflightVariant,
    alpha: float,
    beta: float,
    gamma: float,
    thresholds: Sequence[float],
    margin_tolerance: float,
    p0a_diagnostics: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    mode = str(variant.p0e_baseline)
    residual_fn = _direct_update_residual if mode == "direct_update_c" else _direct_align_residual
    radius = split_conformal_radius([residual_fn(row) for row in coverage_rows], alpha=alpha)
    coverage_audit = [
        {
            "microbatch_id": str(row.get("microbatch_id", "unknown")),
            "residual_radius": residual_fn(row),
            "deployed_radius": radius,
            "covered": bool(residual_fn(row) <= radius),
        }
        for row in coverage_rows
    ]
    if not np.isfinite(radius):
        selected_threshold = None
        threshold_summary = None
        certificates = [_rejected_certificate(row, threshold=0.0) for row in evaluation_rows]
        fail_closed_reason = f"{mode}_radius_is_infinite"
    else:
        risk_materialized = _materialize_static_certificates(
            risk_rows,
            thresholds,
            lambda row, threshold: _direct_certificate(
                row,
                mode=mode,
                radius=radius,
                threshold=threshold,
                margin_tolerance=margin_tolerance,
            ),
        )
        selected = select_risk_threshold(risk_materialized, thresholds, beta=beta, gamma=gamma)
        selected_threshold = selected.selected_threshold
        threshold_summary = {
            "selected_threshold": selected.selected_threshold,
            "threshold_rows": selected.threshold_rows,
            "risk_rows": risk_materialized,
            "beta": beta,
            "gamma": gamma,
            "margin_tolerance": margin_tolerance,
        }
        certificates = (
            [_rejected_certificate(row, threshold=0.0) for row in evaluation_rows]
            if selected_threshold is None
            else [
                _direct_certificate(
                    row,
                    mode=mode,
                    radius=radius,
                    threshold=selected_threshold,
                    margin_tolerance=margin_tolerance,
                )
                for row in evaluation_rows
            ]
        )
        fail_closed_reason = None
    return _finalize_variant(
        variant=variant,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        thresholds=thresholds,
        grid_size=1,
        coverage_rows=coverage_rows,
        risk_rows=risk_rows,
        evaluation_rows=evaluation_rows,
        certificates=certificates,
        coverage_audit=coverage_audit,
        selected_threshold=selected_threshold,
        calibrated_radius=radius,
        deployed_radius=radius,
        threshold_summary=threshold_summary,
        p0a_diagnostics=p0a_diagnostics,
        margin_tolerance=margin_tolerance,
        extra_metrics={"fail_closed_reason": fail_closed_reason},
    )


def _run_matched_random_dropout(
    coverage_rows: Sequence[Mapping[str, Any]],
    risk_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    *,
    variant: PreflightVariant,
    alpha: float,
    matched_acceptance_rate: float | None,
    seed: int,
    margin_tolerance: float,
    p0a_diagnostics: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    target = 0.5 if matched_acceptance_rate is None else float(matched_acceptance_rate)
    target = min(1.0, max(0.0, target))
    certificates = [
        _random_dropout_certificate(row, acceptance_rate=target, seed=seed)
        for row in evaluation_rows
    ]
    coverage_audit = [
        {
            "microbatch_id": str(row.get("microbatch_id", "unknown")),
            "residual_radius": _coefficient_l2_residual(row),
            "deployed_radius": float("nan"),
            "covered": True,
        }
        for row in coverage_rows
    ]
    return _finalize_variant(
        variant=variant,
        alpha=alpha,
        beta=float("nan"),
        gamma=float("nan"),
        thresholds=[0.0],
        grid_size=1,
        coverage_rows=coverage_rows,
        risk_rows=risk_rows,
        evaluation_rows=evaluation_rows,
        certificates=certificates,
        coverage_audit=coverage_audit,
        selected_threshold=0.0,
        calibrated_radius=float("nan"),
        deployed_radius=float("nan"),
        threshold_summary=None,
        p0a_diagnostics=p0a_diagnostics,
        margin_tolerance=margin_tolerance,
        extra_metrics={
            "matched_acceptance_rate": target,
            "matched_acceptance_source": "default_0.5" if matched_acceptance_rate is None else "setgrpo_full",
        },
    )


def _run_anchor_to_nominal_recovery_audit(
    coverage_rows: Sequence[Mapping[str, Any]],
    risk_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    *,
    variant: PreflightVariant,
    alpha: float,
    beta: float,
    gamma: float,
    thresholds: Sequence[float],
    grid_size: int,
    margin_tolerance: float,
    p0a_diagnostics: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Audit the PDF-defined safe-anchor -> nominal-GRPO recovery path.

    This is intentionally audit-only: accepted certificates are still produced
    by the same rule as full SetGRPO.  The extra diagnostics answer whether the
    predefined path can recover non-zero eta before changing the deployed
    method.
    """

    calibrated_radius = conformal_radius_from_rows(coverage_rows, alpha=alpha)
    deployed_radius = calibrated_radius
    coverage_audit = _coverage_audit(coverage_rows, deployed_radius)
    prepared_risk = [_with_radius(row, deployed_radius) for row in risk_rows]
    prepared_evaluation = [_with_radius(row, deployed_radius) for row in evaluation_rows]

    threshold_summary: dict[str, Any] | None = None
    fail_closed_reason: str | None = None
    if not np.isfinite(deployed_radius):
        selected_threshold = None
        fail_closed_reason = "split_conformal_radius_is_infinite; calibration split is too small for requested alpha"
    else:
        threshold_summary = select_threshold_from_microbatches(
            prepared_risk,
            thresholds,
            beta=beta,
            gamma=gamma,
            grid_size=grid_size,
            margin_tolerance=margin_tolerance,
        )
        selected = threshold_summary.get("selected_threshold")
        selected_threshold = None if selected is None else float(selected)

    evaluation_path_audit = _anchor_to_nominal_path_audit(
        prepared_evaluation,
        grid_size=grid_size,
        threshold=0.0 if selected_threshold is None else selected_threshold,
        margin_tolerance=margin_tolerance,
    )
    audit_by_id = {str(row.get("microbatch_id", "unknown")): row for row in evaluation_path_audit}

    certificates = (
        [_rejected_certificate(row, threshold=0.0) for row in prepared_evaluation]
        if selected_threshold is None
        else [
            certify_microbatch(
                row,
                threshold=selected_threshold,
                grid_size=grid_size,
                margin_tolerance=margin_tolerance,
            ).to_dict()
            for row in prepared_evaluation
        ]
    )
    for certificate in certificates:
        row_id = str(certificate.get("microbatch_id", "unknown"))
        if row_id in audit_by_id:
            certificate["anchor_to_nominal_path"] = audit_by_id[row_id]

    risk_path_audit = _anchor_to_nominal_path_audit(
        prepared_risk,
        grid_size=grid_size,
        threshold=0.0,
        margin_tolerance=margin_tolerance,
    )
    return _finalize_variant(
        variant=variant,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        thresholds=thresholds,
        grid_size=grid_size,
        coverage_rows=coverage_rows,
        risk_rows=risk_rows,
        evaluation_rows=evaluation_rows,
        certificates=certificates,
        coverage_audit=coverage_audit,
        selected_threshold=selected_threshold,
        calibrated_radius=calibrated_radius,
        deployed_radius=deployed_radius,
        threshold_summary=threshold_summary,
        p0a_diagnostics=p0a_diagnostics,
        margin_tolerance=margin_tolerance,
        extra_metrics={
            "fail_closed_reason": fail_closed_reason,
            "anchor_to_nominal_path_audit": {
                "risk_threshold0": _summarize_path_audit(risk_path_audit),
                "evaluation_selected_threshold": _summarize_path_audit(evaluation_path_audit),
            },
        },
    )


def _run_support_margin_decomposition_audit(
    coverage_rows: Sequence[Mapping[str, Any]],
    risk_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    *,
    variant: PreflightVariant,
    alpha: float,
    beta: float,
    gamma: float,
    thresholds: Sequence[float],
    grid_size: int,
    margin_tolerance: float,
    p0a_diagnostics: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Decompose why support certificates reject nominal progress.

    This run is diagnostic only.  It keeps the same calibrated ellipsoid and
    risk-selection protocol as full SetGRPO, then records the support-margin
    terms at the safe anchor and nominal GRPO point.
    """

    calibrated_radius = conformal_radius_from_rows(coverage_rows, alpha=alpha)
    deployed_radius = calibrated_radius
    coverage_audit = _coverage_audit(coverage_rows, deployed_radius)
    prepared_risk = [_with_radius(row, deployed_radius) for row in risk_rows]
    prepared_evaluation = [_with_radius(row, deployed_radius) for row in evaluation_rows]

    threshold_summary: dict[str, Any] | None = None
    fail_closed_reason: str | None = None
    if not np.isfinite(deployed_radius):
        selected_threshold = None
        fail_closed_reason = "split_conformal_radius_is_infinite; calibration split is too small for requested alpha"
    else:
        threshold_summary = select_threshold_from_microbatches(
            prepared_risk,
            thresholds,
            beta=beta,
            gamma=gamma,
            grid_size=grid_size,
            margin_tolerance=margin_tolerance,
        )
        selected = threshold_summary.get("selected_threshold")
        selected_threshold = None if selected is None else float(selected)

    risk_decomposition = _support_margin_decomposition_audit(prepared_risk)
    evaluation_decomposition = _support_margin_decomposition_audit(prepared_evaluation)
    decomposition_by_id = {
        str(row.get("microbatch_id", "unknown")): row for row in evaluation_decomposition
    }

    certificates = [_rejected_certificate(row, threshold=0.0) for row in prepared_evaluation]
    for certificate in certificates:
        row_id = str(certificate.get("microbatch_id", "unknown"))
        if row_id in decomposition_by_id:
            certificate["support_margin_decomposition"] = decomposition_by_id[row_id]

    return _finalize_variant(
        variant=variant,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        thresholds=thresholds,
        grid_size=grid_size,
        coverage_rows=coverage_rows,
        risk_rows=risk_rows,
        evaluation_rows=evaluation_rows,
        certificates=certificates,
        coverage_audit=coverage_audit,
        selected_threshold=selected_threshold,
        calibrated_radius=calibrated_radius,
        deployed_radius=deployed_radius,
        threshold_summary=threshold_summary,
        p0a_diagnostics=p0a_diagnostics,
        margin_tolerance=margin_tolerance,
        extra_metrics={
            "fail_closed_reason": fail_closed_reason,
            "support_margin_decomposition_audit": {
                "risk": _summarize_support_margin_decomposition(risk_decomposition),
                "evaluation": _summarize_support_margin_decomposition(evaluation_decomposition),
            },
        },
    )


def _run_radius_scale_sweep_audit(
    coverage_rows: Sequence[Mapping[str, Any]],
    risk_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    *,
    variant: PreflightVariant,
    alpha: float,
    beta: float,
    gamma: float,
    thresholds: Sequence[float],
    grid_size: int,
    margin_tolerance: float,
    radius_scales: Sequence[float],
    p0a_diagnostics: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Sweep scaled conformal radii to diagnose over-conservative penalties."""

    calibrated_radius = conformal_radius_from_rows(coverage_rows, alpha=alpha)
    base_coverage_audit = _coverage_audit(coverage_rows, calibrated_radius)
    scale_results: list[dict[str, Any]] = []
    selected_scale_result: dict[str, Any] | None = None

    for radius_scale in radius_scales:
        deployed_radius = float(calibrated_radius * radius_scale)
        coverage_audit = _coverage_audit(coverage_rows, deployed_radius)
        prepared_risk = [_with_radius(row, deployed_radius) for row in risk_rows]
        prepared_evaluation = [_with_radius(row, deployed_radius) for row in evaluation_rows]

        if not np.isfinite(deployed_radius):
            threshold_summary = None
            selected_threshold = None
            certificates = [_rejected_certificate(row, threshold=0.0) for row in prepared_evaluation]
            fail_closed_reason = "scaled_split_conformal_radius_is_infinite"
        else:
            threshold_summary = select_threshold_from_microbatches(
                prepared_risk,
                thresholds,
                beta=beta,
                gamma=gamma,
                grid_size=grid_size,
                margin_tolerance=margin_tolerance,
            )
            selected = threshold_summary.get("selected_threshold")
            selected_threshold = None if selected is None else float(selected)
            certificates = (
                [_rejected_certificate(row, threshold=0.0) for row in prepared_evaluation]
                if selected_threshold is None
                else [
                    certify_microbatch(
                        row,
                        threshold=selected_threshold,
                        grid_size=grid_size,
                        margin_tolerance=margin_tolerance,
                    ).to_dict()
                    for row in prepared_evaluation
                ]
            )
            fail_closed_reason = None

        metrics = _summarize_certificates(certificates, prepared_evaluation)
        empirical_coverage = (
            float(np.mean([row["covered"] for row in coverage_audit])) if coverage_audit else 0.0
        )
        selected_for_path = 0.0 if selected_threshold is None else selected_threshold
        evaluation_path_audit = _anchor_to_nominal_path_audit(
            prepared_evaluation,
            grid_size=grid_size,
            threshold=selected_for_path,
            margin_tolerance=margin_tolerance,
        )
        evaluation_decomposition = _support_margin_decomposition_audit(prepared_evaluation)

        result = {
            "radius_scale": float(radius_scale),
            "deployed_radius": deployed_radius,
            "coverage": empirical_coverage,
            "coverage_gap": empirical_coverage - (1.0 - alpha),
            "selected_threshold": selected_threshold,
            "fail_closed_reason": fail_closed_reason,
            "risk_selection": _compact_threshold_summary(threshold_summary),
            "evaluation_metrics": metrics,
            "evaluation_path_audit": _summarize_path_audit(evaluation_path_audit),
            "evaluation_support_margin_decomposition": _summarize_support_margin_decomposition(
                evaluation_decomposition
            ),
        }
        scale_results.append(result)
        if selected_scale_result is None and selected_threshold is not None and metrics["accepted_rows"] > 0:
            selected_scale_result = result

    certificates = [_rejected_certificate(row, threshold=0.0) for row in evaluation_rows]
    final_info, certificates, coverage_audit = _finalize_variant(
        variant=variant,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        thresholds=thresholds,
        grid_size=grid_size,
        coverage_rows=coverage_rows,
        risk_rows=risk_rows,
        evaluation_rows=evaluation_rows,
        certificates=certificates,
        coverage_audit=base_coverage_audit,
        selected_threshold=None,
        calibrated_radius=calibrated_radius,
        deployed_radius=calibrated_radius,
        threshold_summary=None,
        p0a_diagnostics=p0a_diagnostics,
        margin_tolerance=margin_tolerance,
        extra_metrics={
            "radius_scale_sweep_audit": {
                "radius_scales": [float(value) for value in radius_scales],
                "first_nonzero_scale": (
                    None if selected_scale_result is None else selected_scale_result["radius_scale"]
                ),
                "scale_results": scale_results,
            },
            "audit_only": True,
        },
    )
    final_info["config"]["radius_scales"] = [float(value) for value in radius_scales]
    return final_info, certificates, coverage_audit


def _with_radius(row: Mapping[str, Any], radius: float) -> dict[str, Any]:
    return {**row, "radius": float(radius)}


def _with_spherical_shape(row: Mapping[str, Any], radius: float) -> dict[str, Any]:
    coefficients = np.asarray(row["estimated_coefficients"], dtype=np.float64)
    return {**row, "shape": np.eye(coefficients.size, dtype=np.float64).tolist(), "radius": float(radius)}


def _finalize_variant(
    *,
    variant: PreflightVariant,
    alpha: float,
    beta: float,
    gamma: float,
    thresholds: Sequence[float],
    grid_size: int,
    coverage_rows: Sequence[Mapping[str, Any]],
    risk_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    certificates: list[dict[str, Any]],
    coverage_audit: list[dict[str, Any]],
    selected_threshold: float | None,
    calibrated_radius: float,
    deployed_radius: float,
    threshold_summary: dict[str, Any] | None,
    p0a_diagnostics: Mapping[str, Any],
    margin_tolerance: float,
    extra_metrics: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = _summarize_certificates(certificates, evaluation_rows)
    empirical_coverage = float(np.mean([row["covered"] for row in coverage_audit])) if coverage_audit else 0.0
    metrics.update(
        {
            "calibrated_radius": float(calibrated_radius),
            "deployed_radius": float(deployed_radius),
            "coverage": empirical_coverage,
            "coverage_target": 1.0 - alpha,
            "coverage_gap": empirical_coverage - (1.0 - alpha),
            "selected_threshold": selected_threshold,
        }
    )
    if extra_metrics:
        metrics.update(dict(extra_metrics))
    final_info = {
        "variant": asdict(variant),
        "config": {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "thresholds": [float(value) for value in thresholds],
            "grid_size": grid_size,
            "margin_tolerance": margin_tolerance,
        },
        "split_sizes": {
            "coverage": len(coverage_rows),
            "risk": len(risk_rows),
            "evaluation": len(evaluation_rows),
        },
        "metrics": metrics,
        "risk_selection": threshold_summary,
        "p0_diagnostics": {
            "p0_a": dict(p0a_diagnostics),
            "p0_gate_report": _p0_gate_report(metrics, p0a_diagnostics, variant_name=variant.name),
        },
    }
    return final_info, certificates, coverage_audit


def _geometry(mapping: Mapping[str, Any]) -> LocalGeometry:
    return LocalGeometry(
        G=np.asarray(mapping["G"], dtype=np.float64),
        P=np.asarray(mapping["P"], dtype=np.float64),
        h=np.asarray(mapping["h"], dtype=np.float64),
    )


def _ellipsoid(row: Mapping[str, Any]) -> CoefficientEllipsoid:
    return CoefficientEllipsoid(
        center=np.asarray(row["estimated_coefficients"], dtype=np.float64),
        shape=np.asarray(row["shape"], dtype=np.float64),
        radius=float(row["radius"]),
    )


def _baseline_certificate(row: Mapping[str, Any]) -> dict[str, Any]:
    geometry = _geometry(row["geometry"])
    coefficients = np.asarray(row["estimated_coefficients"], dtype=np.float64)
    update = geometry.update(coefficients)
    reversed_value = _reversed_against_reference(row, geometry, update)
    return {
        "microbatch_id": str(row.get("microbatch_id", "unknown")),
        "threshold": 0.0,
        "accepted": True,
        "eta_star": 1.0,
        "coefficients": coefficients.tolist(),
        "full_margin": None,
        "reward_margin": None,
        "reversed_against_reference": reversed_value,
        "update_norm": float(np.linalg.norm(update)),
    }


def _safe_anchor_only_certificate(
    row: Mapping[str, Any],
    *,
    threshold: float,
    margin_tolerance: float = 0.0,
) -> dict[str, Any]:
    try:
        ellipsoid = _ellipsoid(row)
        geometry = _geometry(row["geometry"])
        reward_geometry = _geometry(row.get("reward_geometry", row["geometry"]))
        coefficients, update = safe_anchor(ellipsoid, geometry)
        reward_update = reward_geometry.update(coefficients)
        full_margin = support_margin(update, ellipsoid, geometry)
        reward_margin = support_margin(reward_update, ellipsoid, reward_geometry)
        accepted = passes_margin_test(
            full_margin,
            reward_margin,
            threshold=threshold,
            margin_tolerance=margin_tolerance,
            require_reward_progress=True,
        )
        if not accepted:
            return _rejected_certificate(row, threshold=threshold)
        return {
            "microbatch_id": str(row.get("microbatch_id", "unknown")),
            "threshold": threshold,
            "accepted": True,
            "eta_star": 0.0,
            "coefficients": coefficients.tolist(),
            "full_margin": float(full_margin),
            "reward_margin": float(reward_margin),
            "reversed_against_reference": _reversed_against_reference(row, geometry, update),
            "update_norm": float(np.linalg.norm(update)),
            "error": None,
        }
    except (SetGRPOError, ValueError) as exc:
        certificate = _rejected_certificate(row, threshold=threshold)
        certificate["error"] = str(exc)
        return certificate


def _bonferroni_certificate(
    row: Mapping[str, Any],
    *,
    radii: np.ndarray,
    threshold: float,
    margin_tolerance: float = 0.0,
) -> dict[str, Any]:
    geometry = _geometry(row["geometry"])
    reward_geometry = _geometry(row.get("reward_geometry", row["geometry"]))
    coefficients = np.asarray(row["estimated_coefficients"], dtype=np.float64)
    update = geometry.update(coefficients)
    reward_update = reward_geometry.update(coefficients)
    full_margin = _hyperrectangle_support_margin(update, coefficients, radii, geometry)
    reward_margin = _hyperrectangle_support_margin(reward_update, coefficients, radii, reward_geometry)
    accepted = passes_margin_test(
        full_margin,
        reward_margin,
        threshold=threshold,
        margin_tolerance=margin_tolerance,
        require_reward_progress=True,
    )
    if not accepted:
        return _rejected_certificate(row, threshold=threshold)
    return {
        "microbatch_id": str(row.get("microbatch_id", "unknown")),
        "threshold": threshold,
        "accepted": True,
        "eta_star": 1.0,
        "coefficients": coefficients.tolist(),
        "full_margin": float(full_margin),
        "reward_margin": float(reward_margin),
        "reversed_against_reference": _reversed_against_reference(row, geometry, update),
        "update_norm": float(np.linalg.norm(update)),
        "error": None,
    }


def _direct_certificate(
    row: Mapping[str, Any],
    *,
    mode: str,
    radius: float,
    threshold: float,
    margin_tolerance: float = 0.0,
) -> dict[str, Any]:
    geometry = _geometry(row["geometry"])
    coefficients = np.asarray(row["estimated_coefficients"], dtype=np.float64)
    update = geometry.update(coefficients)
    if mode == "direct_update_c":
        margin = float(np.dot(update, update) - radius * np.linalg.norm(update))
    elif mode == "direct_align_c":
        margin = float(np.dot(update, update) - radius)
    else:
        raise ValueError(f"unknown direct conformal mode {mode!r}")
    accepted = margin + margin_tolerance >= threshold
    if not accepted:
        return _rejected_certificate(row, threshold=threshold)
    return {
        "microbatch_id": str(row.get("microbatch_id", "unknown")),
        "threshold": threshold,
        "accepted": True,
        "eta_star": 1.0,
        "coefficients": coefficients.tolist(),
        "full_margin": margin,
        "reward_margin": margin,
        "reversed_against_reference": _reversed_against_reference(row, geometry, update),
        "update_norm": float(np.linalg.norm(update)),
        "error": None,
    }


def _random_dropout_certificate(row: Mapping[str, Any], *, acceptance_rate: float, seed: int) -> dict[str, Any]:
    row_id = str(row.get("microbatch_id", row.get("query_root", "unknown")))
    draw = _stable_unit_interval(f"{seed}:{row_id}")
    if draw > acceptance_rate:
        return _rejected_certificate(row, threshold=0.0)
    return _baseline_certificate(row)


def _certify_variant(
    row: Mapping[str, Any],
    *,
    threshold: float,
    grid_size: int,
    use_recovery: bool,
    require_reward_progress: bool,
    margin_tolerance: float = 0.0,
) -> dict[str, Any]:
    if use_recovery:
        # Reuse the audited recovery path, but evaluate the reward constraint
        # separately for the no-reward-constraint ablation.
        candidates = np.linspace(0.0, 1.0, grid_size)
        ellipsoid = _ellipsoid(row)
        full_geometry = _geometry(row["geometry"])
        reward_geometry = _geometry(row.get("reward_geometry", row["geometry"]))
        from setgrpo_preflight.core import safe_anchor

        anchor, _ = safe_anchor(ellipsoid, full_geometry)
    else:
        candidates = np.asarray([1.0], dtype=np.float64)
        ellipsoid = _ellipsoid(row)
        full_geometry = _geometry(row["geometry"])
        reward_geometry = _geometry(row.get("reward_geometry", row["geometry"]))
        anchor = ellipsoid.center

    best: dict[str, Any] | None = None
    for eta in candidates:
        coefficients = (1.0 - eta) * anchor + eta * ellipsoid.center
        update = full_geometry.update(coefficients)
        reward_update = reward_geometry.update(coefficients)
        full_margin = support_margin(update, ellipsoid, full_geometry)
        reward_margin = support_margin(reward_update, ellipsoid, reward_geometry)
        accepted = passes_margin_test(
            full_margin,
            reward_margin,
            threshold=threshold,
            margin_tolerance=margin_tolerance,
            require_reward_progress=require_reward_progress,
        )
        if accepted:
            best = {
                "microbatch_id": str(row.get("microbatch_id", "unknown")),
                "threshold": threshold,
                "accepted": True,
                "eta_star": float(eta),
                "coefficients": coefficients.tolist(),
                "full_margin": float(full_margin),
                "reward_margin": float(reward_margin),
                "reversed_against_reference": _reversed_against_reference(row, full_geometry, update),
                "update_norm": float(np.linalg.norm(update)),
            }
    return best if best is not None else _rejected_certificate(row, threshold=threshold)


def _anchor_to_nominal_path_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    grid_size: int,
    threshold: float,
    margin_tolerance: float,
) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for row in rows:
        row_id = str(row.get("microbatch_id", "unknown"))
        try:
            ellipsoid = _ellipsoid(row)
            full_geometry = _geometry(row["geometry"])
            reward_geometry = _geometry(row.get("reward_geometry", row["geometry"]))
            anchor_coefficients, anchor_update = safe_anchor(ellipsoid, full_geometry)
            nominal_coefficients = ellipsoid.center
            nominal_update = full_geometry.update(nominal_coefficients)

            best_eta: float | None = None
            best_eta_full_only: float | None = None
            best_eta_reward_only: float | None = None
            best_update_norm: float | None = None
            best_full_margin: float | None = None
            best_reward_margin: float | None = None
            max_full_margin: float | None = None
            max_reward_margin: float | None = None
            anchor_full_margin: float | None = None
            anchor_reward_margin: float | None = None
            nominal_full_margin: float | None = None
            nominal_reward_margin: float | None = None

            for eta in np.linspace(0.0, 1.0, grid_size):
                coefficients = (1.0 - eta) * anchor_coefficients + eta * nominal_coefficients
                update = full_geometry.update(coefficients)
                reward_update = reward_geometry.update(coefficients)
                full_margin = support_margin(update, ellipsoid, full_geometry)
                reward_margin = support_margin(reward_update, ellipsoid, reward_geometry)
                max_full_margin = (
                    full_margin if max_full_margin is None else max(max_full_margin, full_margin)
                )
                max_reward_margin = (
                    reward_margin if max_reward_margin is None else max(max_reward_margin, reward_margin)
                )
                if eta == 0.0:
                    anchor_full_margin = full_margin
                    anchor_reward_margin = reward_margin
                if eta == 1.0:
                    nominal_full_margin = full_margin
                    nominal_reward_margin = reward_margin

                full_ok = _passes_full_margin_only(
                    full_margin,
                    threshold=threshold,
                    margin_tolerance=margin_tolerance,
                )
                reward_ok = _passes_reward_margin_only(
                    reward_margin,
                    margin_tolerance=margin_tolerance,
                )
                if full_ok:
                    best_eta_full_only = float(eta)
                if reward_ok:
                    best_eta_reward_only = float(eta)
                if passes_margin_test(
                    full_margin,
                    reward_margin,
                    threshold=threshold,
                    margin_tolerance=margin_tolerance,
                    require_reward_progress=True,
                ):
                    best_eta = float(eta)
                    best_update_norm = float(np.linalg.norm(update))
                    best_full_margin = float(full_margin)
                    best_reward_margin = float(reward_margin)

            audits.append(
                {
                    "microbatch_id": row_id,
                    "threshold": float(threshold),
                    "margin_tolerance": float(margin_tolerance),
                    "eta_star_path": best_eta,
                    "eta_star_full_margin_only": best_eta_full_only,
                    "eta_star_reward_margin_only": best_eta_reward_only,
                    "anchor_update_norm": float(np.linalg.norm(anchor_update)),
                    "nominal_update_norm": float(np.linalg.norm(nominal_update)),
                    "best_update_norm": best_update_norm,
                    "anchor_full_margin": _none_or_float(anchor_full_margin),
                    "anchor_reward_margin": _none_or_float(anchor_reward_margin),
                    "nominal_full_margin": _none_or_float(nominal_full_margin),
                    "nominal_reward_margin": _none_or_float(nominal_reward_margin),
                    "best_full_margin": best_full_margin,
                    "best_reward_margin": best_reward_margin,
                    "max_full_margin": _none_or_float(max_full_margin),
                    "max_reward_margin": _none_or_float(max_reward_margin),
                    "error": None,
                }
            )
        except (SetGRPOError, ValueError, np.linalg.LinAlgError) as exc:
            audits.append(
                {
                    "microbatch_id": row_id,
                    "threshold": float(threshold),
                    "margin_tolerance": float(margin_tolerance),
                    "eta_star_path": None,
                    "eta_star_full_margin_only": None,
                    "eta_star_reward_margin_only": None,
                    "anchor_update_norm": None,
                    "nominal_update_norm": None,
                    "best_update_norm": None,
                    "anchor_full_margin": None,
                    "anchor_reward_margin": None,
                    "nominal_full_margin": None,
                    "nominal_reward_margin": None,
                    "best_full_margin": None,
                    "best_reward_margin": None,
                    "max_full_margin": None,
                    "max_reward_margin": None,
                    "error": str(exc),
                }
            )
    return audits


def _passes_full_margin_only(
    full_margin: float,
    *,
    threshold: float,
    margin_tolerance: float,
) -> bool:
    return (
        full_margin >= threshold
        if margin_tolerance <= 0.0
        else full_margin + margin_tolerance >= threshold
    )


def _passes_reward_margin_only(reward_margin: float, *, margin_tolerance: float) -> bool:
    return reward_margin > 0.0 if margin_tolerance <= 0.0 else reward_margin + margin_tolerance >= 0.0


def _summarize_path_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("error") is None]
    eta_values = _finite_values(row.get("eta_star_path") for row in valid)
    full_only_eta_values = _finite_values(row.get("eta_star_full_margin_only") for row in valid)
    reward_only_eta_values = _finite_values(row.get("eta_star_reward_margin_only") for row in valid)
    return {
        "rows": len(rows),
        "valid_rows": len(valid),
        "error_rows": len(rows) - len(valid),
        "eta_path": _finite_stats(eta_values),
        "eta_full_margin_only": _finite_stats(full_only_eta_values),
        "eta_reward_margin_only": _finite_stats(reward_only_eta_values),
        "eta_positive_rows": int(np.sum(eta_values > 0.0)) if eta_values.size else 0,
        "eta_zero_rows": int(np.sum(eta_values == 0.0)) if eta_values.size else 0,
        "eta_missing_rows": len(valid) - int(eta_values.size),
        "nominal_reached_rows": int(np.sum(eta_values >= 1.0)) if eta_values.size else 0,
        "full_margin_only_positive_rows": (
            int(np.sum(full_only_eta_values > 0.0)) if full_only_eta_values.size else 0
        ),
        "reward_margin_only_positive_rows": (
            int(np.sum(reward_only_eta_values > 0.0)) if reward_only_eta_values.size else 0
        ),
        "anchor_update_norm": _finite_stats(row.get("anchor_update_norm") for row in valid),
        "nominal_update_norm": _finite_stats(row.get("nominal_update_norm") for row in valid),
        "best_update_norm": _finite_stats(row.get("best_update_norm") for row in valid),
        "anchor_full_margin": _finite_stats(row.get("anchor_full_margin") for row in valid),
        "anchor_reward_margin": _finite_stats(row.get("anchor_reward_margin") for row in valid),
        "nominal_full_margin": _finite_stats(row.get("nominal_full_margin") for row in valid),
        "nominal_reward_margin": _finite_stats(row.get("nominal_reward_margin") for row in valid),
        "max_full_margin": _finite_stats(row.get("max_full_margin") for row in valid),
        "max_reward_margin": _finite_stats(row.get("max_reward_margin") for row in valid),
    }


def _support_margin_decomposition_audit(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for row in rows:
        row_id = str(row.get("microbatch_id", "unknown"))
        try:
            ellipsoid = _ellipsoid(row)
            full_geometry = _geometry(row["geometry"])
            reward_geometry = _geometry(row.get("reward_geometry", row["geometry"]))
            anchor_coefficients, anchor_update = safe_anchor(ellipsoid, full_geometry)
            nominal_coefficients = ellipsoid.center
            nominal_update = full_geometry.update(nominal_coefficients)
            anchor_reward_update = reward_geometry.update(anchor_coefficients)
            nominal_reward_update = reward_geometry.update(nominal_coefficients)
            audits.append(
                {
                    "microbatch_id": row_id,
                    "radius": float(ellipsoid.radius),
                    "coefficient_dim": int(ellipsoid.center.size),
                    "shape_condition": float(np.linalg.cond(ellipsoid.shape)),
                    "shape_trace": float(np.trace(ellipsoid.shape)),
                    "anchor_update_norm": float(np.linalg.norm(anchor_update)),
                    "nominal_update_norm": float(np.linalg.norm(nominal_update)),
                    "anchor_full": _support_margin_decomposition(
                        anchor_update,
                        ellipsoid,
                        full_geometry,
                    ),
                    "anchor_reward": _support_margin_decomposition(
                        anchor_reward_update,
                        ellipsoid,
                        reward_geometry,
                    ),
                    "nominal_full": _support_margin_decomposition(
                        nominal_update,
                        ellipsoid,
                        full_geometry,
                    ),
                    "nominal_reward": _support_margin_decomposition(
                        nominal_reward_update,
                        ellipsoid,
                        reward_geometry,
                    ),
                    "error": None,
                }
            )
        except (SetGRPOError, ValueError, np.linalg.LinAlgError) as exc:
            audits.append(
                {
                    "microbatch_id": row_id,
                    "radius": None,
                    "coefficient_dim": None,
                    "shape_condition": None,
                    "shape_trace": None,
                    "anchor_update_norm": None,
                    "nominal_update_norm": None,
                    "anchor_full": None,
                    "anchor_reward": None,
                    "nominal_full": None,
                    "nominal_reward": None,
                    "error": str(exc),
                }
            )
    return audits


def _support_margin_decomposition(
    direction: np.ndarray,
    ellipsoid: CoefficientEllipsoid,
    geometry: LocalGeometry,
) -> dict[str, float]:
    direction = np.asarray(direction, dtype=np.float64)
    center_update = geometry.update(ellipsoid.center)
    nominal_term = float(np.dot(direction, center_update))
    pulled_back = geometry.G.T @ (geometry.P.T @ direction)
    penalty_sq = float(pulled_back.T @ ellipsoid.shape @ pulled_back)
    penalty_norm = float(np.sqrt(max(0.0, penalty_sq)))
    radius_penalty = float(ellipsoid.radius * penalty_norm)
    margin = float(nominal_term - radius_penalty)
    abs_nominal = abs(nominal_term)
    return {
        "direction_norm": float(np.linalg.norm(direction)),
        "center_update_norm": float(np.linalg.norm(center_update)),
        "nominal_term": nominal_term,
        "pulled_back_l2_norm": float(np.linalg.norm(pulled_back)),
        "penalty_norm": penalty_norm,
        "radius": float(ellipsoid.radius),
        "radius_penalty": radius_penalty,
        "margin": margin,
        "support_ratio": float(nominal_term / radius_penalty) if radius_penalty > 0.0 else float("inf"),
        "penalty_to_abs_nominal": (
            float(radius_penalty / abs_nominal) if abs_nominal > 0.0 else float("inf")
        ),
    }


def _summarize_support_margin_decomposition(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("error") is None]
    return {
        "rows": len(rows),
        "valid_rows": len(valid),
        "error_rows": len(rows) - len(valid),
        "radius": _finite_stats(row.get("radius") for row in valid),
        "shape_condition": _finite_stats(row.get("shape_condition") for row in valid),
        "shape_trace": _finite_stats(row.get("shape_trace") for row in valid),
        "anchor_update_norm": _finite_stats(row.get("anchor_update_norm") for row in valid),
        "nominal_update_norm": _finite_stats(row.get("nominal_update_norm") for row in valid),
        "anchor_full": _summarize_decomposition_point(valid, "anchor_full"),
        "anchor_reward": _summarize_decomposition_point(valid, "anchor_reward"),
        "nominal_full": _summarize_decomposition_point(valid, "nominal_full"),
        "nominal_reward": _summarize_decomposition_point(valid, "nominal_reward"),
    }


def _summarize_decomposition_point(
    rows: Sequence[Mapping[str, Any]],
    point: str,
) -> dict[str, Any]:
    margins = _finite_values(_decomposition_value(row, point, "margin") for row in rows)
    nominal_terms = _finite_values(_decomposition_value(row, point, "nominal_term") for row in rows)
    radius_penalties = _finite_values(
        _decomposition_value(row, point, "radius_penalty") for row in rows
    )
    penalty_to_abs_nominal = _finite_values(
        _decomposition_value(row, point, "penalty_to_abs_nominal") for row in rows
    )
    return {
        "direction_norm": _finite_stats(
            _decomposition_value(row, point, "direction_norm") for row in rows
        ),
        "nominal_term": _finite_stats(nominal_terms),
        "pulled_back_l2_norm": _finite_stats(
            _decomposition_value(row, point, "pulled_back_l2_norm") for row in rows
        ),
        "penalty_norm": _finite_stats(
            _decomposition_value(row, point, "penalty_norm") for row in rows
        ),
        "radius_penalty": _finite_stats(radius_penalties),
        "margin": _finite_stats(margins),
        "support_ratio": _finite_stats(
            _decomposition_value(row, point, "support_ratio") for row in rows
        ),
        "penalty_to_abs_nominal": _finite_stats(penalty_to_abs_nominal),
        "margin_positive_rows": int(np.sum(margins > 0.0)) if margins.size else 0,
        "penalty_exceeds_abs_nominal_rows": (
            int(np.sum(radius_penalties > np.abs(nominal_terms)))
            if radius_penalties.size and nominal_terms.size == radius_penalties.size
            else 0
        ),
    }


def _decomposition_value(row: Mapping[str, Any], point: str, key: str) -> object | None:
    value = row.get(point)
    if not isinstance(value, Mapping):
        return None
    return value.get(key)


def _compact_threshold_summary(summary: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {
        "selected_threshold": summary.get("selected_threshold"),
        "threshold_rows": summary.get("threshold_rows", []),
        "beta": summary.get("beta"),
        "gamma": summary.get("gamma"),
        "margin_tolerance": summary.get("margin_tolerance"),
    }


def _finite_stats(values: Iterable[object] | np.ndarray) -> dict[str, float | int | None]:
    arr = values if isinstance(values, np.ndarray) else _finite_values(values)
    if arr.size == 0:
        return {"count": 0, "min": None, "p50": None, "p90": None, "max": None, "mean": None}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p50": float(np.quantile(arr, 0.5)),
        "p90": float(np.quantile(arr, 0.9)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _finite_values(values: Iterable[object]) -> np.ndarray:
    return np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(float(value))],
        dtype=np.float64,
    )


def _none_or_float(value: object) -> float | None:
    return None if value is None else float(value)


def _rejected_certificate(row: Mapping[str, Any], *, threshold: float) -> dict[str, Any]:
    certificate = MicrobatchCertificate(
        microbatch_id=str(row.get("microbatch_id", "unknown")),
        threshold=threshold,
        accepted=False,
        eta_star=0.0,
        coefficients=None,
        full_margin=None,
        reward_margin=None,
        reversed_against_reference=None,
        update_norm=None,
    ).to_dict()
    certificate["coefficients"] = [0.0 for _ in row["estimated_coefficients"]]
    return certificate


def _reversed_against_reference(
    row: Mapping[str, Any], geometry: LocalGeometry, update: np.ndarray
) -> bool | None:
    reference = row.get("reference_coefficients")
    if reference is None:
        return None
    reference_update = geometry.update(reference)
    return bool(float(np.dot(update, reference_update)) < 0.0)


def _coefficient_delta(row: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(row["reference_coefficients"], dtype=np.float64) - np.asarray(
        row["estimated_coefficients"], dtype=np.float64
    )


def _coefficient_l2_residual(row: Mapping[str, Any]) -> float:
    return float(np.linalg.norm(_coefficient_delta(row)))


def _bonferroni_radii(rows: Sequence[Mapping[str, Any]], *, alpha: float) -> np.ndarray:
    first = np.asarray(rows[0]["estimated_coefficients"], dtype=np.float64)
    dim = first.size
    per_coordinate_alpha = alpha / max(dim, 1)
    radii = []
    for index in range(dim):
        residuals = [abs(float(_coefficient_delta(row)[index])) for row in rows]
        radii.append(split_conformal_radius(residuals, alpha=per_coordinate_alpha))
    return np.asarray(radii, dtype=np.float64)


def _hyperrectangle_support_margin(
    direction: np.ndarray,
    center: np.ndarray,
    radii: np.ndarray,
    geometry: LocalGeometry,
) -> float:
    nominal = float(np.dot(direction, geometry.update(center)))
    pulled_back = geometry.G.T @ (geometry.P.T @ direction)
    penalty = float(np.sum(np.abs(pulled_back) * radii))
    return nominal - penalty


def _direct_update_residual(row: Mapping[str, Any]) -> float:
    geometry = _geometry(row["geometry"])
    estimated = geometry.update(row["estimated_coefficients"])
    reference = geometry.update(row["reference_coefficients"])
    return float(np.linalg.norm(reference - estimated))


def _direct_align_residual(row: Mapping[str, Any]) -> float:
    geometry = _geometry(row["geometry"])
    estimated = geometry.update(row["estimated_coefficients"])
    reference = geometry.update(row["reference_coefficients"])
    # Nonconformity is how much the reference alignment falls below the
    # nominal self-alignment.  This gives a deploy-time lower bound
    # dot(u_hat, u_ref) >= dot(u_hat, u_hat) - q_alpha.
    return float(max(0.0, np.dot(estimated, estimated) - np.dot(estimated, reference)))


def _materialize_static_certificates(
    rows: Sequence[Mapping[str, Any]],
    thresholds: Sequence[float],
    certify,
) -> list[dict[str, Any]]:
    materialized: list[dict[str, Any]] = []
    for row in rows:
        for threshold in thresholds:
            cert = certify(row, float(threshold))
            materialized.append(
                {
                    **cert,
                    "reversed": bool(cert.get("reversed_against_reference"))
                    if cert.get("reversed_against_reference") is not None
                    else False,
                    "yield": float(cert.get("eta_star", 0.0)) if bool(cert.get("accepted")) else 0.0,
                }
            )
    return materialized


def _p0a_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    usable = []
    finite_difference_residuals = []
    linearization_residuals = []
    amplification = []
    nominal_reversals = 0
    for row in rows:
        metadata = row.get("metadata", {})
        if not isinstance(metadata, Mapping):
            continue
        estimated_scores = metadata.get("estimated_scores")
        reference_scores = metadata.get("reference_scores")
        if estimated_scores is None or reference_scores is None:
            continue
        try:
            estimated = np.asarray(estimated_scores, dtype=np.float64)
            reference = np.asarray(reference_scores, dtype=np.float64)
            actual, linearized = linearized_standardization_error(estimated, reference)
            raw_delta = reference - estimated
            linearization_residuals.append(float(np.linalg.norm(actual - linearized)))
            amplification.append(float(np.linalg.norm(linearized) / max(np.linalg.norm(raw_delta), 1e-12)))
            finite_difference_residuals.append(
                _finite_difference_jacobian_error(estimated, reference - estimated)
            )
            geometry = _geometry(row["geometry"])
            nominal = geometry.update(row["estimated_coefficients"])
            ref = geometry.update(row["reference_coefficients"])
            nominal_reversals += int(float(np.dot(nominal, ref)) < 0.0)
            usable.append(row)
        except Exception:
            continue
    return {
        "usable_rows": len(usable),
        "mean_finite_difference_jacobian_error": _finite_mean(finite_difference_residuals),
        "mean_linearization_residual": _finite_mean(linearization_residuals),
        "mean_standardization_amplification": _finite_mean(amplification),
        "nominal_reference_reversal_rate": nominal_reversals / len(usable) if usable else None,
        "status": "measured" if usable else "missing_raw_score_metadata",
    }


def _finite_difference_jacobian_error(scores: np.ndarray, perturbation: np.ndarray) -> float:
    eps = 1e-6
    analytic = group_standardization_jacobian(scores) @ perturbation
    finite = (
        group_standardized_coefficients(scores + eps * perturbation)
        - group_standardized_coefficients(scores - eps * perturbation)
    ) / (2 * eps)
    return float(np.linalg.norm(analytic - finite))


def _p0_gate_report(
    metrics: Mapping[str, Any],
    p0a_diagnostics: Mapping[str, Any],
    *,
    variant_name: str,
) -> dict[str, Any]:
    coverage_gap = float(metrics.get("coverage_gap", float("nan")))
    acceptance_rate = float(metrics.get("acceptance_rate", 0.0))
    reversal_rate = metrics.get("reversal_rate")
    selected_threshold = metrics.get("selected_threshold")
    p0a_ok = (
        p0a_diagnostics.get("status") == "measured"
        and (p0a_diagnostics.get("mean_finite_difference_jacobian_error") or 0.0) < 1e-4
    )
    p0b_ok = bool(np.isfinite(coverage_gap) and coverage_gap >= -0.05)
    p0c_ok = bool(acceptance_rate > 0.0 and metrics.get("mean_eta_star") is not None)
    p0d_ok = bool(selected_threshold is not None and acceptance_rate > 0.0)
    p0e_status = "baseline_run" if variant_name not in {"setgrpo_full", "grpo"} else "requires_cross_variant_comparison"
    if reversal_rate is None:
        risk_ok = "not_auditable_without_reference"
    else:
        risk_ok = bool(float(reversal_rate) <= 0.05)
    return {
        "p0_a_phenomenon_derivative_gate": "pass" if p0a_ok else "blocked",
        "p0_b_joint_coverage_gate": "pass" if p0b_ok else "blocked",
        "p0_c_recovery_gate": "pass" if p0c_ok else "blocked",
        "p0_d_selective_risk_gate": "pass" if p0d_ok else "blocked",
        "p0_e_decomposition_gate": p0e_status,
        "audited_directional_risk_ok": risk_ok,
        "strict_p0_ready_for_real_rl": bool(p0a_ok and p0b_ok and p0c_ok and p0d_ok and risk_ok is True),
    }


def _stable_unit_interval(value: str) -> float:
    import hashlib

    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _summarize_certificates(
    certificates: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    accepted = [row for row in certificates if bool(row["accepted"])]
    audited = [row for row in accepted if row.get("reversed_against_reference") is not None]
    coefficient_drifts: list[float] = []
    for certificate, source in zip(certificates, rows):
        coefficients = certificate.get("coefficients")
        if coefficients is None:
            continue
        coefficient_drifts.append(
            float(
                np.linalg.norm(
                    np.asarray(coefficients, dtype=np.float64)
                    - np.asarray(source["estimated_coefficients"], dtype=np.float64)
                )
            )
        )

    return {
        "evaluation_rows": len(certificates),
        "accepted_rows": len(accepted),
        "acceptance_rate": len(accepted) / len(certificates),
        "reversal_count": sum(bool(row["reversed_against_reference"]) for row in audited),
        "reversal_rate": (
            sum(bool(row["reversed_against_reference"]) for row in audited) / len(audited)
            if audited
            else None
        ),
        "mean_eta_star": _finite_mean(row.get("eta_star") for row in accepted),
        "mean_update_norm": _finite_mean(row.get("update_norm") for row in accepted),
        "mean_full_margin": _finite_mean(row.get("full_margin") for row in accepted),
        "mean_reward_margin": _finite_mean(row.get("reward_margin") for row in accepted),
        "mean_coefficient_drift": _finite_mean(coefficient_drifts),
    }


def _finite_mean(values: Iterable[object]) -> float | None:
    materialized = np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(float(value))],
        dtype=np.float64,
    )
    return None if materialized.size == 0 else float(np.mean(materialized))
