"""Framework-independent SetGRPO reference implementation."""

from .calibration import CoefficientSet, JointRewardSetCalibration, joint_coefficients, split_conformal_radius
from .geometry import LocalPolicyGeometry
from .recovery import GeometryGuidedSafeRecovery, RecoveryResult
from .risk import SelectiveRiskGatedOptimization, RiskCalibrationResult
from .online import SetGRPOUpdate, build_update
from .certificate import FrozenCertificateProvider, certify_row, freeze_rule
from .warmup import build_frozen_artifacts
from .training import attach_coefficients, prepare_grpo_batch

__all__ = [
    "CoefficientSet",
    "JointRewardSetCalibration",
    "joint_coefficients",
    "split_conformal_radius",
    "LocalPolicyGeometry",
    "GeometryGuidedSafeRecovery",
    "RecoveryResult",
    "SelectiveRiskGatedOptimization",
    "RiskCalibrationResult",
    "SetGRPOUpdate",
    "build_update",
    "FrozenCertificateProvider",
    "certify_row",
    "freeze_rule",
    "build_frozen_artifacts",
    "attach_coefficients",
    "prepare_grpo_batch",
]
