"""Runtime patch activated only with SETGRPO_JACOBIAN_SIDECAR=1."""
from __future__ import annotations

import os

if os.getenv("SETGRPO_JACOBIAN_SIDECAR") == "1":
    import ray
    from verl import DataProto
    from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.utils.fsdp_utils import get_fsdp_full_state_dict
    from verl.single_controller.base.decorator import Dispatch, register
    from .jacobian_service import JacobianSidecar

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def _export_state(self, data):
        import torch.distributed as dist
        state = get_fsdp_full_state_dict(self.actor_module_fsdp, offload_to_cpu=True, rank0_only=True)
        return DataProto.from_dict(meta_info={"setgrpo_actor_state": state if dist.get_rank() == 0 else {}})

    ActorRolloutRefWorker.export_setgrpo_actor_state = _export_state

    _original_fit = RayDAPOTrainer.fit

    def _fit_with_sidecar(self, *args, **kwargs):
        model_path = str(self.config.actor_rollout_ref.model.path)
        self._setgrpo_jacobian_sidecar = JacobianSidecar.options(name="setgrpo-jacobian-sidecar").remote(
            model_path,
            int(os.getenv("SETGRPO_JACOBIAN_PROJECTION_DIM", "256")),
            os.getenv("SETGRPO_JACOBIAN_PARAMETER_PATTERN", ".*norm.*weight$"),
            int(os.getenv("SETGRPO_JACOBIAN_SEED", "2026")),
        )
        original = self.actor_rollout_wg.compute_setgrpo_jacobian

        def sidecar_compute(data):
            state_payload = self.actor_rollout_wg.export_setgrpo_actor_state(DataProto.from_dict())
            state = state_payload.meta_info.get("setgrpo_actor_state", {})
            if state:
                ray.get(self._setgrpo_jacobian_sidecar.sync.remote(state))
            payload = data.to("cpu")
            result = ray.get(self._setgrpo_jacobian_sidecar.compute.remote(payload))
            return DataProto.from_dict(tensors=result)

        self.actor_rollout_wg.compute_setgrpo_jacobian = sidecar_compute
        print("[SetGRPO] Jacobian sidecar enabled on dedicated GPU", flush=True)
        try:
            return _original_fit(self, *args, **kwargs)
        finally:
            self.actor_rollout_wg.compute_setgrpo_jacobian = original
            ray.kill(self._setgrpo_jacobian_sidecar)

    RayDAPOTrainer.fit = _fit_with_sidecar
