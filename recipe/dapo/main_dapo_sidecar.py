"""Opt-in formal-training entrypoint using one dedicated Ray Jacobian GPU.

The normal ``recipe.dapo.main_dapo`` entrypoint is untouched.  This module
patches only the trainer instance method for this experimental launcher.
"""
from __future__ import annotations

import os

import ray

from verl import DataProto
from .dapo_ray_trainer import RayDAPOTrainer


def _install_state_export():
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.utils.fsdp_utils import get_fsdp_full_state_dict
    import torch.distributed as dist
    from verl.single_controller.base.decorator import Dispatch, register

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def export_setgrpo_actor_state(self, data):
        state = get_fsdp_full_state_dict(self.actor_module_fsdp, offload_to_cpu=True, rank0_only=True)
        return DataProto.from_dict(meta_info={"setgrpo_actor_state": state if dist.get_rank() == 0 else {}})

    ActorRolloutRefWorker.export_setgrpo_actor_state = export_setgrpo_actor_state


def _install_sidecar() -> None:
    from setgrpo_sidecar.jacobian_service import JacobianSidecar

    _install_state_export()
    original_fit = RayDAPOTrainer.fit

    def fit(self, *args, **kwargs):
        sidecar = JacobianSidecar.options(name=f"setgrpo-jacobian-{os.getpid()}").remote(
            str(self.config.actor_rollout_ref.model.path),
            int(os.getenv("SETGRPO_JACOBIAN_PROJECTION_DIM", "256")),
            os.getenv("SETGRPO_JACOBIAN_PARAMETER_PATTERN", ".*norm.*weight$"),
            int(os.getenv("SETGRPO_JACOBIAN_SEED", "2026")),
        )
        original_compute = self.actor_rollout_wg.compute_setgrpo_jacobian

        def compute(data):
            state_payload = self.actor_rollout_wg.export_setgrpo_actor_state(DataProto.from_dict())
            state = state_payload.meta_info.get("setgrpo_actor_state", {})
            if state:
                ray.get(sidecar.sync.remote(state))
            result = ray.get(sidecar.compute.remote(data.to("cpu")))
            return DataProto.from_dict(tensors=result)

        self.actor_rollout_wg.compute_setgrpo_jacobian = compute
        print("[SetGRPO] dedicated Jacobian sidecar enabled", flush=True)
        try:
            return original_fit(self, *args, **kwargs)
        finally:
            self.actor_rollout_wg.compute_setgrpo_jacobian = original_compute
            ray.kill(sidecar)

    RayDAPOTrainer.fit = fit


_install_sidecar()

# The two-node Slurm launcher creates the Ray cluster before starting this
# process. Connect here so main_dapo.run_ppo does not create a local cluster.
if not ray.is_initialized() and os.getenv("RAY_ADDRESS"):
    ray.init(
        address=os.environ["RAY_ADDRESS"],
        runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}},
    )

from .main_dapo import main  # noqa: E402

if __name__ == "__main__":
    main()
