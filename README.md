# SetGRPO

This repository contains a small, framework-independent reference implementation of **Set-Valued Group Relative Policy Optimization (SetGRPO)**.

SetGRPO keeps the original GRPO objective and adds three certified components:

1. **Joint Reward Set Calibration** builds a joint coefficient uncertainty set from paired reward observations and split-conformal calibration.
2. **Geometry-Guided Safe Recovery** uses local policy geometry to recover the largest update that satisfies the support and reward-progress constraints.
3. **Selective Risk-Gated Optimization** chooses a deployment threshold with a finite-sample upper bound on direction-reversal risk.

This repository contains the complete paper training source tree: the SetGRPO calibration modules, policy-Jacobian path, online certificate provider, GRPO advantage hook, Ray/FSDP trainer, rollout workers, configuration, reward adapter, and external-judge client. It does not ship model weights, private datasets, API keys, checkpoints, or a cluster-specific submission script.

## Installation

```bash
pip install -e .
# For GPU training integration:
pip install -e '.[training]'
# For the full paper training runtime:
pip install -e '.[verl]'
```

The core implementation requires Python 3.10+, NumPy, and SciPy. PyTorch is optional and is only needed for projected policy-Jacobian extraction or tensor-batch attachment.

## Minimal workflow

```python
import numpy as np
from setgrpo.calibration import CoefficientSet, joint_coefficients, split_conformal_radius
from setgrpo.geometry import LocalPolicyGeometry
from setgrpo.recovery import GeometryGuidedSafeRecovery
from setgrpo.risk import SelectiveRiskGatedOptimization

estimated = joint_coefficients([0.2, 0.7, 0.4, 0.1])
reference = joint_coefficients([0.3, 0.8, 0.2, 0.0])
shape = np.eye(4) * 0.05
radius = split_conformal_radius([1.1, 0.8, 1.4, 0.9], alpha=0.1)
coefficient_set = CoefficientSet(estimated, shape, radius)

geometry = LocalPolicyGeometry(
    response_jacobian=np.eye(4),
    projection=np.eye(4),
    offset=np.zeros(4),
)
recovery = GeometryGuidedSafeRecovery(coefficient_set, geometry)
result = recovery.recover(threshold=0.0)
print(result.accepted, result.coefficients)
```

For online GRPO, use `setgrpo.warmup.build_frozen_artifacts` during startup with paired scores and projected response Jacobians. Load the resulting `FrozenCertificateProvider`, then call `setgrpo.training.prepare_grpo_batch` once per rollout batch. The downstream GRPO advantage implementation must consume `batch['setgrpo_coefficients']`; this package provides that adapter but does not pretend to replace Verl's model rollout and optimizer runtime.

## Package layout

- `setgrpo.calibration`: group normalization, coefficient sets, covariance shapes, and conformal radii.
- `setgrpo.geometry`: local policy geometry and optional projected response Jacobians.
- `setgrpo.recovery`: support margins, safe anchors, and geometry-guided recovery.
- `setgrpo.risk`: selective risk calibration and threshold selection.
- `setgrpo.online`: framework-neutral online GRPO update construction.
- `setgrpo.data`: JSON-friendly certificate row helpers.
- `setgrpo.warmup` and `setgrpo.certificate`: paired-score calibration, frozen risk rules, and deploy-time certificates.
- `setgrpo.training`: minimal tensor-batch adapter; no trainer framework is required.

The caller owns model rollout, grader calls, and response-level Jacobians. Integrations with Verl, vLLM, RubricHub, or other systems can wrap the minimal training adapter without copying their private infrastructure.

## Tests

```bash
pytest -q
```

The public source checkout is runnable after installation for the synthetic
calibration/update examples:

```bash
cd setgrpo-open-source
pip install -e '.[test]'
python examples/offline_certificate.py
python examples/online_update.py
```

The training command below is the paper entrypoint. It expects a local model,
an application-provided parquet dataset, and `STAGE2_API_KEY` when the external
judge is enabled:

```bash
python -m recipe.dapo.main_dapo \
  actor_rollout_ref.model.path=/path/to/model \
  data.train_files=/path/to/train.parquet \
  algorithm.adv_estimator=setgrpo \
  algorithm.setgrpo.online.enabled=true \
  algorithm.setgrpo.online.provider_type=online_certificate \
  custom_reward_function.path=health_bench/scaleai_batch_reward_fn.py \
  custom_reward_function.name=compute_score_batched \
  trainer.n_gpus_per_node=4 trainer.nnodes=1
```

Use `recipe/dapo/run_setgrpo_rubrichub_external_api.sh` as the environment
wrapper when using an OpenAI-compatible external grader. SetGRPO requires the
caller to provide the model/data paths and API key; none are embedded here.

## License

Apache-2.0. See `LICENSE`.
