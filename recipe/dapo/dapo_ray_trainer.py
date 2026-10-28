# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, RayPPOTrainer, apply_kl_penalty, compute_advantage, compute_response_mask
from verl.utils.debug import marked_timer

from setgrpo_preflight.certificate_provider import CertificateCoefficientProvider
from setgrpo_preflight.core import SetGRPOError
from setgrpo_preflight.online_certificate import OnlineCertificateProvider
from setgrpo_preflight.online import attach_online_coefficients, build_online_step, load_coefficient_provider
from setgrpo_preflight.warmup import build_warmup_rows


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def _online_setgrpo_enabled(self) -> bool:
        """Return whether this run explicitly enables online SetGRPO wiring."""

        return bool(
            self.config.algorithm.adv_estimator == AdvantageEstimator.SETGRPO
            and self.config.algorithm.get("setgrpo", {}).get("online", {}).get("enabled", False)
        )

    def _online_setgrpo_config(self):
        return self.config.algorithm.get("setgrpo", {}).get("online", {})

    @staticmethod
    def _json_safe(value):
        """Convert trainer values to strict JSON without leaking NaN/Inf."""

        if isinstance(value, torch.Tensor):
            return RayDAPOTrainer._json_safe(value.detach().cpu().tolist())
        if isinstance(value, np.ndarray):
            return RayDAPOTrainer._json_safe(value.tolist())
        if isinstance(value, np.generic):
            return RayDAPOTrainer._json_safe(value.item())
        if isinstance(value, Mapping):
            return {str(key): RayDAPOTrainer._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [RayDAPOTrainer._json_safe(item) for item in value]
        if isinstance(value, float):
            return value if np.isfinite(value) else None
        if isinstance(value, (str, int, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def _append_jsonl(path: Path, row: Mapping) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(RayDAPOTrainer._json_safe(row), ensure_ascii=False, allow_nan=False) + "\n")

    def _init_artifact_paths(self) -> None:
        """Initialize optional local artifacts used by curves and appendices."""

        trainer_cfg = self.config.trainer
        metrics_dir = trainer_cfg.get("metrics_data_dir")
        diagnostics_dir = trainer_cfg.get("setgrpo_diagnostics_dir")
        jacobian_dir = trainer_cfg.get("setgrpo_jacobian_data_dir")
        self._metrics_path = Path(str(metrics_dir)).expanduser() / "training_metrics.jsonl" if metrics_dir else None
        self._diagnostics_path = Path(str(diagnostics_dir)).expanduser() / "setgrpo_diagnostics.jsonl" if diagnostics_dir else None
        self._jacobian_dir = Path(str(jacobian_dir)).expanduser() if jacobian_dir else None
        self._last_setgrpo_diagnostics = []
        # Running aggregates used by the paper's training curves.  These are
        # deliberately kept in the trainer process so they are also available
        # when the per-group diagnostics are written to JSONL.
        self._setgrpo_cumulative_recovery = 0.0
        self._setgrpo_cumulative_nonzero_updates = 0
        self._setgrpo_seen_groups = 0
        for path in (self._metrics_path, self._diagnostics_path):
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
        if self._jacobian_dir is not None:
            self._jacobian_dir.mkdir(parents=True, exist_ok=True)

    def _persist_training_metrics(self, step: int, metrics: Mapping) -> None:
        if self._metrics_path is None:
            return
        row = {"step": int(step)}
        row.update(metrics)
        self._append_jsonl(self._metrics_path, row)

    def _persist_setgrpo_diagnostics(self, step: int) -> None:
        if self._diagnostics_path is None:
            return
        for diagnostic in self._last_setgrpo_diagnostics:
            self._append_jsonl(self._diagnostics_path, {"step": int(step), **diagnostic})

    def _persist_jacobian_snapshot(self, step: int, batch: DataProto) -> None:
        """Persist the current-step projected Jacobian for geometry audits."""

        if self._jacobian_dir is None or "setgrpo_jacobian" not in batch.batch:
            return
        jacobian = batch.batch["setgrpo_jacobian"].detach().float().cpu().numpy()
        group_ids = np.asarray([str(value) for value in batch.non_tensor_batch.get("uid", [])], dtype=str)
        target = self._jacobian_dir / f"step_{int(step):06d}.npz"
        np.savez_compressed(target, step=np.asarray([int(step)], dtype=np.int64), jacobian=jacobian, group_ids=group_ids)

    def _get_online_setgrpo_provider(self):
        if hasattr(self, "_online_setgrpo_provider_loaded"):
            return self._online_setgrpo_provider
        online_cfg = self._online_setgrpo_config()
        path = online_cfg.get("provider_path")
        name = online_cfg.get("provider_name", "build_coefficients")
        provider_type = str(online_cfg.get("provider_type", "module"))
        if provider_type == "online_certificate":
            certificate_path = online_cfg.get("certificate_path")
            if not certificate_path:
                raise RuntimeError("online_certificate provider requires setgrpo.online.certificate_path")
            self._online_setgrpo_provider = OnlineCertificateProvider(
                certificate_path,
                rule_path=online_cfg.get("rule_path"),
                key=str(online_cfg.get("certificate_key", "microbatch_id")),
                grid_size=int(online_cfg.get("grid_size", 257)),
                variant=str(online_cfg.get("variant", "full")),
                discard_rate=float(online_cfg.get("discard_rate", 0.0)),
            )
        elif provider_type == "certificate":
            certificate_path = online_cfg.get("certificate_path")
            if not certificate_path:
                raise RuntimeError("certificate provider requires setgrpo.online.certificate_path")
            self._online_setgrpo_provider = CertificateCoefficientProvider(
                certificate_path,
                key=str(online_cfg.get("certificate_key", "microbatch_id")),
            )
        elif provider_type == "module":
            self._online_setgrpo_provider = load_coefficient_provider(path, name) if path else None
        else:
            raise RuntimeError(f"unsupported online SetGRPO provider_type: {provider_type}")
        self._online_setgrpo_provider_loaded = True
        return self._online_setgrpo_provider

    def _attach_online_setgrpo_coefficients(self, batch: DataProto, metrics: dict) -> DataProto:
        """Build and attach coefficients for the current rollout batch.

        This is the trainer integration boundary.  Production runs must supply
        a certificate provider; reward-standardized coefficients are available
        only with an explicit diagnostic flag.
        """

        if not self._online_setgrpo_enabled():
            return batch
        if "uid" not in batch.non_tensor_batch:
            raise RuntimeError("online SetGRPO requires batch.non_tensor_batch['uid']")
        rewards = batch.batch["token_level_rewards"].sum(dim=-1).detach().float().cpu().tolist()
        group_ids = [str(value) for value in batch.non_tensor_batch["uid"]]
        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
        metadata = batch.non_tensor_batch.get("extra_info")
        if "setgrpo_jacobian" in batch.batch:
            jacobians = batch.batch["setgrpo_jacobian"].detach().cpu().numpy()
            # ``extra_info`` may be a NumPy array; evaluating it as a boolean
            # raises ``ValueError: truth value of an array is ambiguous``.
            base_metadata = [{}] * len(batch) if metadata is None else metadata
            grouped_indices = defaultdict(list)
            for index, group_id in enumerate(group_ids):
                grouped_indices[str(group_id)].append(index)
            metadata = [dict(item or {}) for item in base_metadata]
            for indices in grouped_indices.values():
                matrix = np.stack([jacobians[index] for index in indices], axis=1).tolist()
                for index in indices:
                    metadata[index]["setgrpo_jacobian"] = matrix
        online_cfg = self._online_setgrpo_config()
        provider = self._get_online_setgrpo_provider()
        # Diagnostics are per-step records.  Clear stale entries before
        # evaluating the current rollout batch.
        if hasattr(provider, "last_diagnostics") and isinstance(provider.last_diagnostics, dict):
            provider.last_diagnostics.clear()
        step = build_online_step(
            rewards,
            group_ids,
            rollout_n=rollout_n,
            coefficient_provider=provider,
            metadata=metadata,
            allow_reward_proxy=bool(online_cfg.get("allow_reward_proxy", False)),
        )
        attach_online_coefficients(batch, step)
        if step.total_groups and step.accepted_groups == 0 and provider is not None:
            raise RuntimeError(
                "online SetGRPO provider matched zero rollout groups; "
                "check certificate_key and rollout metadata"
            )
        metrics["setgrpo/online_groups"] = step.total_groups
        metrics["setgrpo/online_accepted_groups"] = step.accepted_groups
        metrics["setgrpo/online_acceptance_rate"] = step.acceptance_rate
        grouped_scores = defaultdict(list)
        for group_id, score in zip(group_ids, rewards):
            grouped_scores[group_id].append(float(score))
        group_stds = [float(np.std(values)) for values in grouped_scores.values()]
        metrics["setgrpo/reward_mean"] = float(np.mean(rewards)) if rewards else 0.0
        metrics["setgrpo/reward_std"] = float(np.std(rewards)) if rewards else 0.0
        if "response_mask" in batch.batch:
            response_lengths = batch.batch["response_mask"].detach().float().sum(dim=-1)
            metrics["setgrpo/response_length_mean"] = float(response_lengths.mean().item())
            metrics["setgrpo/response_length_std"] = float(response_lengths.std(unbiased=False).item())
        metrics["setgrpo/group_reward_std_mean"] = float(np.mean(group_stds)) if group_stds else 0.0
        metrics["setgrpo/coefficient_l2"] = float(np.linalg.norm(np.asarray(step.coefficients, dtype=np.float64)))
        metrics["setgrpo/online_rejected_groups"] = metrics["setgrpo/online_groups"] - metrics["setgrpo/online_accepted_groups"]
        diagnostics = getattr(provider, "last_diagnostics", {})
        self._last_setgrpo_diagnostics = [
            {"group_id": str(row_id), **dict(value)} for row_id, value in diagnostics.items()
        ]
        metrics["setgrpo/rejected_groups"] = sum(1 for value in diagnostics.values() if value.get("status") == "rejected")
        accepted_diag = [v for v in diagnostics.values() if v.get("status") == "accepted"]
        eta_values = [
            float(v["eta_star"])
            for v in accepted_diag
            if v.get("eta_star") is not None and np.isfinite(float(v["eta_star"]))
        ]
        nonzero_groups = sum(1 for value in eta_values if abs(value) > 0.0)
        self._setgrpo_cumulative_recovery += float(np.sum(eta_values))
        self._setgrpo_cumulative_nonzero_updates += nonzero_groups
        self._setgrpo_seen_groups += len(diagnostics)
        metrics["setgrpo/accepted_mean_recovery_ratio"] = float(np.mean(eta_values)) if eta_values else 0.0
        metrics["setgrpo/nonzero_update_rate"] = (
            float(nonzero_groups / len(diagnostics)) if diagnostics else 0.0
        )
        metrics["setgrpo/cumulative_recovery"] = self._setgrpo_cumulative_recovery
        metrics["setgrpo/cumulative_nonzero_update_rate"] = (
            float(self._setgrpo_cumulative_nonzero_updates / self._setgrpo_seen_groups)
            if self._setgrpo_seen_groups else 0.0
        )
        for metric_name in ("full_margin", "reward_margin", "eta_star"):
            values = [float(v[metric_name]) for v in accepted_diag if metric_name in v and np.isfinite(float(v[metric_name]))]
            metrics[f"setgrpo/{metric_name}"] = float(np.mean(values)) if values else 0.0
        if "setgrpo_jacobian" in batch.batch:
            jacobian_values = batch.batch["setgrpo_jacobian"].detach().float()
            metrics["setgrpo/jacobian_norm"] = float(torch.linalg.vector_norm(jacobian_values).item())
        metrics["setgrpo/deferred_groups"] = sum(1 for value in diagnostics.values() if value.get("status") == "deferred")
        reason_counts = defaultdict(int)
        for value in diagnostics.values():
            reason_counts[str(value.get("reason", "unknown"))] += 1
        for reason, count in reason_counts.items():
            safe_reason = "".join(ch if ch.isalnum() else "_" for ch in reason).strip("_") or "unknown"
            metrics[f"setgrpo/reason/{safe_reason}"] = int(count)
        rule = getattr(provider, "_rule", None)
        if isinstance(rule, Mapping):
            if rule.get("radius") is not None:
                metrics["setgrpo/frozen_conformal_radius"] = float(rule["radius"])
            if rule.get("selected_threshold") is not None:
                metrics["setgrpo/selected_threshold"] = float(rule["selected_threshold"])
        data_sources = batch.non_tensor_batch.get("data_source")
        if data_sources is not None and len(data_sources) == len(rewards):
            by_domain = defaultdict(list)
            for source, reward in zip(data_sources, rewards):
                by_domain[str(source)].append(float(reward))
            for source, values in by_domain.items():
                safe_source = "".join(ch if ch.isalnum() else "_" for ch in source).strip("_") or "unknown"
                metrics[f"setgrpo/domain_reward/{safe_source}"] = float(np.mean(values))
        if torch.cuda.is_available():
            metrics["system/peak_gpu_memory_gb"] = float(torch.cuda.max_memory_allocated() / (1024 ** 3))
        return batch

    def _run_online_warmup(self) -> None:
        """Freeze a rule from one rollout with two independent grader passes."""
        if not self._online_setgrpo_enabled():
            return
        cfg = self._online_setgrpo_config().get("warmup", {})
        if not bool(cfg.get("enabled", False)):
            return
        if self.use_rm:
            raise RuntimeError("online SetGRPO warmup currently requires reward_fn scoring without a separate reward model")
        output_dir = Path(str(cfg.get("output_dir", "setgrpo_warmup")))
        cached_certificate = output_dir / "certificate_rows.jsonl"
        cached_rule = output_dir / "frozen_rule.json"
        if cached_certificate.is_file() and cached_certificate.stat().st_size > 0 and cached_rule.is_file() and cached_rule.stat().st_size > 0:
            online_cfg = self.config.algorithm.setgrpo.online
            online_cfg.certificate_path = str(cached_certificate)
            online_cfg.rule_path = str(cached_rule)
            self._online_setgrpo_provider_loaded = False
            self._online_setgrpo_provider = None
            print(f"[SetGRPO][warmup] reusing cached certificate from {output_dir}", flush=True)
            return
        warmup_groups = int(os.getenv("SETGRPO_WARMUP_GROUPS", str(self.config.data.train_batch_size)))
        if warmup_groups < 3:
            raise RuntimeError("SETGRPO_WARMUP_GROUPS must be at least 3")
        batches = []
        groups_loaded = 0
        loader = iter(self.train_dataloader)
        while groups_loaded < warmup_groups:
            try:
                batch_dict = next(loader)
            except StopIteration:
                break
            batch = DataProto.from_single_dict(batch_dict)
            batches.append(batch)
            groups_loaded += len(batch)
        if not batches:
            raise RuntimeError("warmup requires at least one training batch")
        base = batches[0] if len(batches) == 1 else DataProto.concat(batches)
        if len(base) > warmup_groups:
            base = base[:warmup_groups]
        pop_tensor = ["input_ids", "attention_mask", "position_ids"]
        pop_meta = ["raw_prompt_ids", "multi_modal_data", "reward_model"]
        gen_batch = base.pop(batch_keys=pop_tensor, non_tensor_batch_keys=[k for k in pop_meta if k in base.non_tensor_batch])
        group_count = len(gen_batch)
        stable_uids = np.asarray([f"warmup-{index}" for index in range(group_count)], dtype=object)
        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
        print(f"[SetGRPO][warmup] start groups={group_count} rollout_n={rollout_n} total_responses={group_count * rollout_n}", flush=True)

        def generate_warmup_rollout(do_sample: bool):
            print(f"[SetGRPO][warmup] rollout start do_sample={do_sample}", flush=True)
            request = deepcopy(gen_batch)
            request.meta_info["do_sample"] = do_sample
            output = self.actor_rollout_wg.generate_sequences(request)
            # Rollout timing is diagnostic metadata and differs on every
            # invocation; remove it before unioning with the source batch.
            output.meta_info.pop("timing", None)
            scored = base.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
            scored.non_tensor_batch["uid"] = np.repeat(stable_uids, self.config.actor_rollout_ref.rollout.n)
            scored = scored.union(output)
            print(f"[SetGRPO][warmup] rollout complete responses={len(scored)}", flush=True)
            return scored

        def score_warmup_rollout(scored: DataProto, label: str):
            print(f"[SetGRPO][warmup] {label}: reward start responses={len(scored)}", flush=True)
            try:
                result = self.reward_fn(scored, return_dict=True)
            except TypeError:
                result = self.reward_fn(scored)
            reward = result["reward_tensor"] if isinstance(result, dict) else result
            values = reward.sum(dim=-1).detach().float()
            print(f"[SetGRPO][warmup] {label}: reward complete responses={len(scored)} mean={values.mean().item():.4f} std={values.std().item():.4f}", flush=True)
            return values.cpu().numpy()

        # The point and reference scores must describe the exact same answer
        # group.  Generating two independent rollouts and pairing them by
        # response position mixes answer-sampling variation into the
        # calibration residual.  Generate once, then independently rescore
        # the frozen responses through the same stochastic grader protocol.
        estimated_batch = generate_warmup_rollout(True)
        reference_scores = score_warmup_rollout(deepcopy(estimated_batch), "reference")
        estimated_scores = score_warmup_rollout(estimated_batch, "estimated")
        estimated_batch.batch["response_mask"] = compute_response_mask(estimated_batch)
        export_path = output_dir / "jacobian_input.pt"
        if os.getenv("SETGRPO_EXPORT_WARMUP_INPUT", "0") == "1":
            torch.save(
                {
                    "input_ids": estimated_batch.batch["input_ids"].cpu(),
                    "attention_mask": estimated_batch.batch["attention_mask"].cpu(),
                    "response_mask": estimated_batch.batch["response_mask"].cpu(),
                    "reference_scores": reference_scores,
                    "estimated_scores": estimated_scores,
                    "rollout_n": rollout_n,
                    "group_ids": stable_uids.tolist(),
                },
                export_path,
            )
            print(f"[SetGRPO][warmup] exported Jacobian input: {export_path}", flush=True)
        if os.getenv("SETGRPO_WARMUP_ONLY", "0") == "1":
            self._warmup_only = True
            return
        print("[SetGRPO][warmup] jacobian start", flush=True)
        jacobian = self.actor_rollout_wg.compute_setgrpo_jacobian(estimated_batch).batch["setgrpo_jacobian"]
        jacobian = jacobian.detach().float().cpu().numpy()
        print(f"[SetGRPO][warmup] jacobian complete shape={jacobian.shape}", flush=True)
        groups = []
        for index, uid in enumerate(stable_uids):
            sl = slice(index * rollout_n, (index + 1) * rollout_n)
            ref_group = reference_scores[sl]
            est_group = estimated_scores[sl]
            if np.std(ref_group) <= 1e-12 or np.std(est_group) <= 1e-12:
                print(
                    f"[SetGRPO][warmup] skipping zero-variance group {uid} "
                    f"reference_std={np.std(ref_group):.3g} estimated_std={np.std(est_group):.3g}",
                    flush=True,
                )
                continue
            groups.append({"microbatch_id": str(uid), "reference_scores": ref_group.tolist(), "estimated_scores": est_group.tolist(), "jacobian": jacobian[sl].T.tolist()})
        if len(groups) < 3:
            raise SetGRPOError(
                f"warmup has only {len(groups)} usable groups after zero-variance filtering; "
                "need at least 3"
            )
        print(f"[SetGRPO][warmup] usable groups={len(groups)}/{len(stable_uids)}", flush=True)
        print(f"[SetGRPO][warmup] freezing certificate output_dir={output_dir}", flush=True)
        # Warmup score covariance is ill-conditioned for small rollout groups.
        # Pass the regularization settings explicitly; relying on a helper
        # default allowed older launchers to disable the condition cap and
        # produced an enormous conformal radius with no selectable threshold.
        result = build_warmup_rows(
            groups,
            output_dir=output_dir,
            alpha=float(cfg.get("alpha", 0.1)),
            beta=float(cfg.get("beta", 0.25)),
            gamma=float(cfg.get("gamma", 0.25)),
            shape_shrinkage=float(cfg.get("shape_shrinkage", 0.1)),
            shape_condition_cap=float(cfg.get("shape_condition_cap", 1000.0)),
            thresholds=[float(x) for x in os.getenv("SETGRPO_WARMUP_THRESHOLDS", "0.0 0.05 0.1 0.2 0.3 0.5").split()],
        )
        online_cfg = self.config.algorithm.setgrpo.online
        online_cfg.certificate_path = result["certificate_path"]
        online_cfg.rule_path = result["rule_path"]
        self._online_setgrpo_provider_loaded = False
        self._online_setgrpo_provider = None
        print(f"[SetGRPO] warmup frozen rule from {result['rows']} groups: {result['rule_path']}", flush=True)
        self._warmup_skip_first_batch = True

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        self._init_artifact_paths()
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # Build the dataset/model-specific frozen rule inside this training
        # job, before the first optimizer update.
        self._run_online_warmup()
        if getattr(self, "_warmup_only", False):
            print("[SetGRPO] warmup-only export complete", flush=True)
            return

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if getattr(self, "_warmup_skip_first_batch", False):
                    # The first dataloader batch was consumed for the paired
                    # warmup rollouts and must not be trained a second time.
                    self._warmup_skip_first_batch = False
                    continue
                do_profile = self.global_steps in (self.config.trainer.profile_steps or [])
                if do_profile:
                    self.actor_rollout_wg.start_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.start_profile()
                    if self.use_critic:
                        self.critic_wg.start_profile()
                    if self.use_rm:
                        self.rm_wg.start_profile()

                metrics = {}

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                
                # Check if graded system prompt feature is enabled, if so, preserve reward_model field
                enable_graded_system_prompt = getattr(self.config.actor_rollout_ref.rollout, 'enable_graded_system_prompt', False)
                if enable_graded_system_prompt and "reward_model" in new_batch.non_tensor_batch:
                    print(f"[DAPO_TRAINER] Graded system prompt feature enabled, passing reward_model info through meta_info")
                    
                    # Extract reward_model information and put it into meta_info as additional data channel
                    reward_models = []
                    for i in range(len(new_batch)):
                        reward_model = new_batch.non_tensor_batch['reward_model'][i]
                        reward_models.append(reward_model)
                    
                    # Store reward_model information in meta_info as backup data channel
                    gen_batch_meta_info = {
                        "graded_system_prompt_reward_models": reward_models,
                        "enable_graded_system_prompt": True
                    }
                    print(f"[DAPO_TRAINER] Successfully extracted {len(reward_models)} reward_models to meta_info")
                    
                    # reward_model can be popped from non_tensor_batch since we're passing it through meta_info
                    non_tensor_batch_keys_to_pop.append("reward_model")
                else:
                    # If graded system prompt feature is not enabled, reward_model field can be popped
                    if "reward_model" in new_batch.non_tensor_batch:
                        non_tensor_batch_keys_to_pop.append("reward_model")
                    gen_batch_meta_info = {}
                
                gen_batch = new_batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
                
                # Add reward_model information to gen_batch's meta_info
                gen_batch.meta_info.update(gen_batch_meta_info)
                
                # Add training step information to meta_info (for step_linear rule)
                gen_batch.meta_info.update({
                    "global_steps": self.global_steps,
                    "total_training_steps": self.total_training_steps
                })

                is_last_step = self.global_steps >= self.total_training_steps

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                with marked_timer("step", timing_raw):
                    print(f"[SetGRPO][step {self.global_steps}/{self.total_training_steps}] generate start prompts={len(gen_batch)} rollout_n={self.config.actor_rollout_ref.rollout.n}", flush=True)
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                    print(f"[SetGRPO][step {self.global_steps}] generate complete trajectories={len(gen_batch_output)}", flush=True)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw, "yellow"):
                        print(f"[SetGRPO][step {self.global_steps}] reward start trajectories={len(new_batch)}", flush=True)
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]
                        reward_values = new_batch.batch["token_level_rewards"].sum(dim=-1).detach().float()
                        print(f"[SetGRPO][step {self.global_steps}] reward complete mean={reward_values.mean().item():.4f} std={reward_values.std().item():.4f}", flush=True)

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = new_batch.batch["token_level_scores"].sum(dim=-1).numpy()

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name]):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [uid for uid, std in prompt_uid2metric_std.items() if std > 0 or len(prompt_uid2metric_vals[uid]) == 1]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                continue
                            else:
                                raise ValueError(f"{num_gen_batches=} >= {max_num_gen_batches=}." + " Generated too many. Please check if your data are too difficult." + " You could also try set max_num_gen_batches=0 to enable endless trials.")
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    # Online SetGRPO coefficients are attached after any group
                    # filtering and immediately before advantage computation.
                    if self._online_setgrpo_enabled():
                        print(f"[SetGRPO][step {self.global_steps}] jacobian start trajectories={len(batch)}", flush=True)
                        batch.meta_info["setgrpo_projection_dim"] = int(self._online_setgrpo_config().get("projection_dim", 256))
                        batch.batch["response_mask"] = compute_response_mask(batch)
                        with marked_timer("jacobian", timing_raw, "purple"):
                            jacobian_output = self.actor_rollout_wg.compute_setgrpo_jacobian(batch)
                        batch.batch["setgrpo_jacobian"] = jacobian_output.batch["setgrpo_jacobian"]
                        print(f"[SetGRPO][step {self.global_steps}] jacobian complete shape={tuple(batch.batch['setgrpo_jacobian'].shape)}", flush=True)
                    batch = self._attach_online_setgrpo_coefficients(batch, metrics)
                    print(f"[SetGRPO][step {self.global_steps}] coefficients complete accepted={metrics.get('setgrpo/online_accepted_groups', 0)}/{metrics.get('setgrpo/online_groups', 0)}", flush=True)

                    # === Updating ===

                    if "response_mask" not in batch.batch:
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )
                    print(f"[SetGRPO][step {self.global_steps}] advantage complete", flush=True)

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        print(f"[SetGRPO][step {self.global_steps}] actor update complete", flush=True)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Persist raw rollout samples when requested.  This keeps
                    # prompt/response/reward/domain records for appendix
                    # analyses instead of relying on the tracking backend.
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, "green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).detach().cpu().tolist()
                            sample_metadata = {
                                key: batch.non_tensor_batch[key]
                                for key in ("uid", "data_source", "split")
                                if key in batch.non_tensor_batch and len(batch.non_tensor_batch[key]) == len(inputs)
                            }
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                                graded_prompt_info=batch.non_tensor_batch.get("graded_prompt_info"),
                                sample_metadata=sample_metadata,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with marked_timer("testing", timing_raw, "green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with marked_timer("save_checkpoint", timing_raw, "green"):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                self._persist_jacobian_snapshot(self.global_steps, batch)
                self._persist_setgrpo_diagnostics(self.global_steps)
                self._persist_training_metrics(self.global_steps, metrics)
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if do_profile:
                    self.actor_rollout_wg.stop_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.stop_profile()
                    if self.use_critic:
                        self.critic_wg.stop_profile()
                    if self.use_rm:
                        self.rm_wg.stop_profile()

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
