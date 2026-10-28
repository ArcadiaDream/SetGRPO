<h1 align="center">SetGRPO: Calibrating Grader-Induced Policy Optimization</h1>

<p align="center">
  <b>Official implementation of Set-Valued Group Relative Policy Optimization</b>
</p>

<p align="center">
  <a href="https://arcadiadream.github.io/HJL.github.io/">Jiale Huang</a><sup>1</sup> &nbsp;·&nbsp;
  <a href="https://lee-zixu.github.io/">Zixu Li</a><sup>1,†</sup> &nbsp;·&nbsp;
  <a href="https://zivchen-ty.github.io/">Zhiwei Chen</a><sup>1</sup> &nbsp;·&nbsp;
  <a href="https://zhihfu.github.io/">Zhiheng Fu</a><sup>1</sup> &nbsp;·&nbsp;
  Jinhe Bi<sup>2</sup> &nbsp;·&nbsp;
  <a href="https://faculty.sdu.edu.cn/huyupeng1/zh_CN/index.htm">Yupeng Hu</a><sup>1,*</sup>
</p>

<p align="center">
  <sup>1</sup>Affiliation of authors 1–4 and 6 &nbsp;&nbsp; <sup>2</sup>Affiliation of author 5<br>
  <sup>†</sup>Project Leader &nbsp;&nbsp; <sup>*</sup>Corresponding Author
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.36807"><img src="https://img.shields.io/badge/arXiv-2610.36807-b31b1b.svg" alt="arXiv"></a>
  <a href="https://arcadiadream.github.io/SetGRPO.github.io/"><img src="https://img.shields.io/badge/Project-Page-orange.svg" alt="Project Page"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green.svg" alt="License"></a>
  <a href="https://github.com/ArcadiaDream/SetGRPO-v2/stargazers"><img src="https://img.shields.io/github/stars/ArcadiaDream/SetGRPO-v2.svg?style=social" alt="Stars"></a>
</p>

---

## 📌 Introduction

Rubric-based reinforcement learning trains large language models on open-ended tasks through fine-grained grader
feedback, but the grader itself is stochastic. Because GRPO normalizes rewards **within each response group** and then
combines the resulting training coefficients with response-level gradients, a single scoring discrepancy can shift the
coefficients of the entire group, and the same coefficient shift can produce a different update direction under a
different policy snapshot. Score discrepancies alone therefore do not describe what the optimizer actually does.

**SetGRPO** keeps the original GRPO objective and treats the within-group training coefficients as a *joint uncertainty
set* instead of a point estimate. It calibrates that set conformally, maps it through the current policy's local update
geometry to recover the largest safe update, and admits a response group only when a globally calibrated threshold
keeps the direction-reversal risk under control. Under exchangeability the construction gives finite-sample marginal
coverage for the joint coefficient set and simultaneous risk control across candidate thresholds.

On ten open-ended benchmarks across three domains (Medical, Writing, Instruction Following), SetGRPO attains the
highest macro average among all compared training methods on both the Qwen3.5 and Llama3.2 model families.

<p align="center">
  <img src="assets/intro-panel-a.svg" alt="Score disagreement propagates jointly" width="46%">
  <img src="assets/intro-panel-b.svg" alt="Policy geometry changes local update alignment" width="46%">
</p>
<p align="center">
  <b>Figure 1.</b> (a) Score disagreement propagates jointly through within-group normalization.
  (b) Policy geometry changes local update alignment.
</p>

## 📢 News

- **[2026-10-27]** Paper released on arXiv: [2610.36807](https://arxiv.org/abs/2610.36807).
- **[2026-10-17]** Training code, preflight tooling, and the Verl/FSDP recipe are public in this repository.

## ✨ Key Features

- 🔍 **Joint Reward Set Calibration** — builds a joint coefficient uncertainty set from paired reward observations,
  with a split-conformal radius that carries finite-sample marginal coverage.
- 🔗 **Geometry-Guided Safe Recovery** — maps candidate coefficients into the current policy's local update geometry
  and returns the largest recovery ratio that keeps the update aligned with the reference direction.
- 🧩 **Selective Risk-Gated Optimization** — selects a single global threshold on an independent risk-calibration set,
  bounding the conditional direction-reversal risk of the updates that survive the gate, while preserving the original
  GRPO objective.
- 🏆 **Consistent gains across backbones** — same-backbone comparison against RubricHub gives **57.56** macro average on
  Qwen3-4B (+1.50) and **60.59** on Qwen3-14B (+1.47), with the joint set and the threshold calibrated separately per
  backbone.

## 🏗️ Architecture

<p align="center">
  <img src="assets/framework.png" alt="Overview of SetGRPO" width="95%">
</p>
<p align="center">
  <b>Figure 2.</b> Overview of SetGRPO, which consists of (a) Joint Reward Set Calibration,
  (b) Geometry-Guided Safe Recovery, and (c) Selective Risk-Gated Optimization.
</p>

1. **Calibrate.** Paired scores from the same grader are normalized within each response group and turned into a joint
   coefficient set with a conformal radius.
2. **Recover.** Each candidate coefficient vector is pushed through the response-level gradients of the current policy;
   the recovery step keeps the largest ratio that still projects positively onto the reference update.
3. **Gate.** A one-sided Clopper–Pearson bound over candidate thresholds yields a single global threshold
   (`τ_sel = 0.25` in the paper), targeting `β = 0.08` direction-reversal risk.
4. **Optimize.** Only groups that pass the gate contribute their recovered coefficients to the original GRPO objective.

## 📊 Experiment Results

Same-backbone comparison against RubricHub on Qwen3-4B and Qwen3-14B. Medical: HB = HealthBench, LEM = LLMEval-Med,
MQA = MedQA, MCQA = MedMCQA. Writing: WB = WritingBench, CW = Creative Writing v3. Instruction Following:
IFE = IFEval, IFB = IFBench, IB = InfoBench, MIF = MulDimIF.

| Backbone | Method | HB | LEM | MQA | MCQA | WB | CW | IFE | IFB | IB | MIF | Macro Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-4B | RubricHub | 64.34 | 45.12 | 78.38 | 61.88 | 66.51 | 28.47 | 64.21 | 24.08 | 80.21 | 47.35 | 56.06 |
| Qwen3-4B | **SetGRPO** | 66.82 | 47.57 | 77.91 | 61.42 | 68.97 | 30.18 | 66.83 | 26.49 | 79.66 | 49.72 | **57.56** |
| Qwen3-14B | RubricHub | 68.44 | 53.36 | 77.53 | 64.76 | 70.42 | 32.71 | 69.74 | 25.61 | 82.88 | 46.79 | 59.12 |
| Qwen3-14B | **SetGRPO** | 70.91 | 54.88 | 76.94 | 64.18 | 72.83 | 34.62 | 72.21 | 27.94 | 82.17 | 49.18 | **60.59** |

Risk calibration at the selected threshold `τ_sel = 0.25`: the one-sided Clopper–Pearson upper bound stays below
`β = 0.08`, with empirical conditional risk **0.0548** on Qwen3-4B and **0.0554** on Qwen3-14B. Full tables, the
risk-calibration curves, and the ablation study are reported in the paper.

## 📑 Table of Contents

- [📌 Introduction](#-introduction)
- [📢 News](#-news)
- [✨ Key Features](#-key-features)
- [🏗️ Architecture](#️-architecture)
- [📊 Experiment Results](#-experiment-results)
- [📂 Repository Structure](#-repository-structure)
- [🚀 Installation](#-installation)
- [📂 Data Preparation](#-data-preparation)
- [🏃 Quick Start](#-quick-start)
- [📦 What Is Not Shipped](#-what-is-not-shipped)
- [🤝 Acknowledgements](#-acknowledgements)
- [✉️ Contact](#️-contact)
- [📝 Citation](#-citation)
- [🔗 Related Projects](#-related-projects)
- [🫡 Support & Contributing](#-support--contributing)
- [📄 License](#-license)

## 📂 Repository Structure

```
SetGRPO/
├── assets/                  # figures referenced by this README
├── setgrpo/                 # framework-independent numerical core
│   ├── calibration.py       # group normalization, coefficient sets, conformal radii
│   ├── geometry.py          # local policy geometry, projected response Jacobians
│   ├── recovery.py          # support margins, safe anchors, geometry-guided recovery
│   ├── risk.py              # selective risk calibration and threshold selection
│   ├── online.py            # framework-neutral online GRPO update construction
│   ├── warmup.py            # paired-score calibration, frozen artifacts
│   ├── certificate.py       # deploy-time certificates
│   ├── training.py          # minimal tensor-batch adapter
│   └── data.py              # JSON-friendly certificate row helpers
├── setgrpo_preflight/       # startup / pre-training checks and diagnostics
├── setgrpo_sidecar/         # policy-Jacobian service
├── recipe/dapo/             # paper training entrypoint (Ray + FSDP + Verl)
├── verl/                    # vendored Verl runtime
├── eval/                    # external-judge API client
├── health_bench/            # batched reward function used in the paper
├── examples/                # synthetic offline / online walkthroughs
└── tests/                   # core unit tests
```

## 🚀 Installation

```bash
git clone https://github.com/ArcadiaDream/SetGRPO-v2.git
cd SetGRPO
```

```bash
pip install -e .                  # numerical core: numpy + scipy only
pip install -e '.[training]'      # GPU integration
pip install -e '.[verl]'          # full paper training runtime
```

The core needs Python 3.10+, NumPy and SciPy. PyTorch is optional and only required for projected policy-Jacobian
extraction or tensor-batch attachment.

## 📂 Data Preparation

This repository does not redistribute any dataset. A training run needs three things from the caller:

**1. Training prompts with rubrics.** A parquet file following the Verl RLHF dataset layout, containing the prompt
field consumed by `data.train_files` and the rubric text used by the batched reward function. The paper trains on the
RubricHub training set; arrange your own copy in the same schema.

**2. A grading endpoint.** Either an OpenAI-compatible external API or a locally served judge. The wrapper reads
`STAGE2_API_BASE_URL`, `STAGE2_API_MODEL`, and `STAGE2_API_KEY` from the environment and never stores credentials.

**3. A local model checkpoint.** Passed as `actor_rollout_ref.model.path`, in the standard Hugging Face layout.

Directory layout used by the recipe:

```
SetGRPO/
├── recipe/dapo/
│   ├── config/dapo_trainer.yaml      # setgrpo.* options live here
│   ├── run_setgrpo_rubrichub_online.sh
│   └── run_setgrpo_rubrichub_external_api.sh
└── <your-data>/
    ├── train.parquet                 # prompts + rubrics
    └── <model>/                      # local checkpoint
```

## 🏃 Quick Start

### Minimal offline certificate

```python
import numpy as np
from setgrpo.calibration import CoefficientSet, joint_coefficients, split_conformal_radius
from setgrpo.geometry import LocalPolicyGeometry
from setgrpo.recovery import GeometryGuidedSafeRecovery

estimated = joint_coefficients([0.2, 0.7, 0.4, 0.1])
shape = np.eye(4) * 0.05
radius = split_conformal_radius([1.1, 0.8, 1.4, 0.9], alpha=0.1)
coefficient_set = CoefficientSet(estimated, shape, radius)

geometry = LocalPolicyGeometry(
    response_jacobian=np.eye(4),
    projection=np.eye(4),
    offset=np.zeros(4),
)
result = GeometryGuidedSafeRecovery(coefficient_set, geometry).recover(threshold=0.0)
print(result.accepted, result.coefficients)
```

### Runnable examples

```bash
pip install -e '.[test]'
python examples/offline_certificate.py
python examples/online_update.py
```

Both scripts use synthetic scores and geometry. They exercise the public calibration and update APIs and do not claim
to reproduce a training run.

### Training

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

Or use the environment wrapper, which preflights the grader endpoint before launching:

```bash
MODEL_PATH=/path/to/model TRAIN_FILE=/path/to/train.parquet \
STAGE2_API_KEY=sk-... \
bash recipe/dapo/run_setgrpo_rubrichub_external_api.sh
```

See [`recipe/dapo/README.md`](recipe/dapo/README.md) for the full runtime requirements.

## 📦 What Is Not Shipped

This repository does not include model weights, private datasets, API keys, checkpoints, or a cluster-specific
launcher. Training requires a caller-provided model, a parquet dataset, and grader credentials.

## 🤝 Acknowledgements

This implementation builds on [Verl](https://github.com/volcengine/verl) for the Ray/FSDP training runtime, and on
conformal prediction and conformal risk control for the statistical guarantees.

## ✉️ Contact

Questions and bug reports are welcome via [Issues](https://github.com/ArcadiaDream/SetGRPO-v2/issues).

## 📝 Citation

```bibtex
@misc{huang2026setgrpo,
  title         = {SetGRPO: Calibrating Grader-Induced Policy Optimization},
  author        = {Huang, Jiale and Li, Zixu and Chen, Zhiwei and Fu, Zhiheng
                   and Bi, Jinhe and Hu, Yupeng},
  year          = {2026},
  eprint        = {2610.36807},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2610.36807}
}
```

If you find this code useful, a ⭐ on the repository is appreciated.

## 🔗 Related Projects

Other works from our group:

- [TEMA (ACL'26)](https://github.com/Lee-zixu/ACL26-TEMA) — anchor the image, follow the text for multi-modification composed image retrieval.
- [ConeSep (CVPR'26)](https://github.com/Lee-zixu/ConeSep) — cone-based robust noise-unlearning compositional network.
- [Air-Know (CVPR'26)](https://github.com/ZhihFu/Air-Know) — arbiter-calibrated knowledge-internalizing robust network.
- [INTENT (AAAI'26)](https://github.com/ZivChen-Ty/INTENT) — invariance and discrimination-aware noise mitigation.
- [HABIT (AAAI'26)](https://github.com/Lee-zixu/HABIT) — chrono-synergia robust progressive learning framework.
- [OFFSET (ACM MM'25)](https://github.com/ZivChen-Ty/OFFSET) — segmentation-based focus shift revision.
- [ENCODER (AAAI'25)](https://github.com/Lee-zixu/ENCODER) — composed image retrieval benchmark suite.

## 🫡 Support & Contributing

We welcome all forms of contributions. If you have a question, an idea, or find a bug:

- Open an [Issue](https://github.com/ArcadiaDream/SetGRPO-v2/issues) for discussions or bug reports.
- Submit a [Pull Request](https://github.com/ArcadiaDream/SetGRPO-v2/pulls) to improve the codebase.

## 📄 License

Apache-2.0. See [LICENSE](LICENSE).
