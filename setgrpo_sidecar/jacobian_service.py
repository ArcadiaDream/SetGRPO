from __future__ import annotations

import re
from typing import Any

import ray
import torch
import torch.nn.functional as F


@ray.remote(num_gpus=1, num_cpus=4)
class JacobianSidecar:
    def __init__(self, model_path: str, projection_dim: int = 256, pattern: str = ".*norm.*weight$", seed: int = 2026):
        from transformers import AutoModelForCausalLM
        from setgrpo_preflight.jacobian import countsketch_project_named_gradients

        self.countsketch = countsketch_project_named_gradients
        self.projection_dim = int(projection_dim)
        self.pattern = pattern
        self.seed = int(seed)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa")
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.enable_input_require_grads()
        self.model.to("cuda")
        self.selected = [(n, p) for n, p in self.model.named_parameters() if re.search(pattern, n)]
        if not self.selected:
            raise RuntimeError(f"no Jacobian parameters match {pattern!r}")
        for n, p in self.model.named_parameters():
            p.requires_grad_(re.search(pattern, n) is not None)

    def sync(self, state):
        # FSDP exports canonical unflattened parameter names on rank zero.
        self.model.load_state_dict(state, strict=True)
        return True

    def compute(self, data: Any):
        local = data.to("cuda")
        columns = []
        self.model.train()
        for index in range(len(local)):
            self.model.zero_grad(set_to_none=True)
            input_ids = local.batch["input_ids"][index:index + 1]
            attention_mask = local.batch["attention_mask"][index:index + 1]
            response_mask = local.batch["response_mask"][index:index + 1]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits[:, :-1, :].float()
            labels = input_ids[:, 1:]
            lp = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            lp = lp[:, -response_mask.shape[1]:]
            mask = response_mask.to(lp.dtype)
            scalar = (lp * mask).sum() / mask.sum().clamp_min(1.0)
            grads = torch.autograd.grad(scalar, [p for _, p in self.selected], allow_unused=True)
            named = [(n, g) for (n, _), g in zip(self.selected, grads) if g is not None]
            columns.append(self.countsketch(named, projection_dim=self.projection_dim, seed=self.seed).detach().float().cpu())
        return {"setgrpo_jacobian": torch.stack(columns, dim=0) if columns else torch.empty((0, self.projection_dim))}
