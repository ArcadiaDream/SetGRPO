import numpy as np

from setgrpo.calibration import CoefficientSet, joint_coefficients, split_conformal_radius
from setgrpo.geometry import LocalPolicyGeometry
from setgrpo.recovery import GeometryGuidedSafeRecovery
from setgrpo.risk import SelectiveRiskGatedOptimization


def test_joint_coefficients_are_centered_and_normalized():
    values = joint_coefficients([1.0, 2.0, 4.0, 8.0])
    assert np.isclose(values.mean(), 0.0)
    assert np.isclose(np.mean(values**2), 1.0)


def test_recovery_returns_a_certified_update():
    coefficient_set = CoefficientSet(np.array([-.8, -.2, .2, .8]), np.eye(4) * 0.01, 0.1)
    geometry = LocalPolicyGeometry(np.eye(4), np.eye(4), np.zeros(4))
    result = GeometryGuidedSafeRecovery(coefficient_set, geometry).recover(threshold=0.0)
    assert result.accepted
    assert result.coefficients is not None


def test_risk_selection_controls_reversal_rate():
    selector = SelectiveRiskGatedOptimization(beta_target=0.2, gamma=0.1)
    result = selector.select(
        [{"threshold": 0.1, "accepted": True, "reversed": False, "yield": 1.0}],
        [0.1],
    )
    assert result.selected_threshold == 0.1


def test_conformal_radius_is_finite_when_rank_is_available():
    assert split_conformal_radius([0.1, 0.2, 0.3], alpha=0.5) == 0.2
