"""Build and inspect one framework-independent SetGRPO certificate."""

import numpy as np

from setgrpo.calibration import CoefficientSet, joint_coefficients, split_conformal_radius
from setgrpo.geometry import LocalPolicyGeometry
from setgrpo.recovery import GeometryGuidedSafeRecovery


def main() -> None:
    center = joint_coefficients([0.2, 0.7, 0.4, 0.1])
    shape = np.eye(4) * 0.05
    # Four calibration values cannot provide the requested 90% finite-sample
    # order statistic; use a conservative finite radius for this smoke run.
    coefficient_set = CoefficientSet(center, shape, split_conformal_radius([0.8, 1.0, 1.1, 0.9], alpha=0.5))
    geometry = LocalPolicyGeometry(np.eye(4), np.eye(4), np.zeros(4))
    result = GeometryGuidedSafeRecovery(coefficient_set, geometry).recover(threshold=0.0)
    print({"accepted": result.accepted, "eta": result.eta})


if __name__ == "__main__":
    main()
