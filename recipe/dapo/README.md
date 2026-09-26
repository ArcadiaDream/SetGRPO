# SetGRPO training recipe

This directory is the paper training entrypoint. It is intentionally kept
separate from the numerical `setgrpo` package so the full runtime can be
installed and audited as a single source checkout.

The recipe requires the vendored `verl` package, Ray, PyTorch, Transformers,
vLLM, and a caller-provided reward function. Start from the repository root:

```bash
pip install -r requirements-paper.txt
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

The external grader wrapper reads `STAGE2_API_BASE_URL`, `STAGE2_API_MODEL`,
and `STAGE2_API_KEY` from the environment. It never stores credentials.
