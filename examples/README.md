# Examples

The examples exercise the public calibration and update APIs with synthetic
scores and geometry. They do not claim to reproduce a model training run.

For a real run, the application must provide:

1. A GRPO rollout backend and model.
2. A reward provider that returns one scalar score per response.
3. Paired calibration scores and response-level policy Jacobians.
4. A GRPO trainer that consumes `setgrpo_coefficients` as token-level
   advantages.

The `.[verl]` extra provides the generic Ray/Transformers/vLLM dependencies,
but the exact Verl version must match the model and CUDA image used by the
application. Private datasets, graders, API keys, and cluster launchers are
intentionally not shipped.
