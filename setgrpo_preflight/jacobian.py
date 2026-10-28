"""Extract projected policy/LoRA response Jacobians for SetGRPO geometry.

For a frozen answer group, column ``i`` of ``G_x`` is a deterministic
CountSketch of ``grad_theta mean_t log pi_theta(o_i,t | q, o_i,<t)`` over the
selected trainable parameters.  This provides an auditable local policy
geometry without materializing a full parameter-by-response Jacobian.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import inspect
import json
import os
import re
from collections.abc import Mapping as MappingABC
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class JacobianExtractionSummary:
    input_rows: int
    enriched_rows: int
    projection_dim: int
    seed: int
    objective: str
    parameter_pattern: str
    selected_parameters: int
    selected_parameter_elements: int


def countsketch_project_named_gradients(
    named_gradients: Iterable[tuple[str, Any]],
    *,
    projection_dim: int,
    seed: int,
    chunk_size: int = 1_000_000,
) -> Any:
    """Project named torch gradients with a deterministic signed CountSketch."""

    import torch

    if projection_dim < 1:
        raise ValueError("projection_dim must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    materialized = [(name, gradient) for name, gradient in named_gradients if gradient is not None]
    if not materialized:
        raise ValueError("at least one non-null gradient is required")
    device = materialized[0][1].device
    output = torch.zeros(projection_dim, device=device, dtype=torch.float32)
    for name, gradient in materialized:
        flat = gradient.detach().reshape(-1).to(device=device, dtype=torch.float32)
        generator = torch.Generator(device=device)
        generator.manual_seed(_parameter_seed(name, seed))
        for start in range(0, flat.numel(), chunk_size):
            values = flat[start : start + chunk_size]
            buckets = torch.randint(
                projection_dim, (values.numel(),), generator=generator, device=device
            )
            signs = torch.randint(2, (values.numel(),), generator=generator, device=device)
            signs = signs.to(torch.float32).mul_(2.0).sub_(1.0)
            output.scatter_add_(0, buckets, values * signs)
    return output


def inject_policy_jacobian_geometry(
    row: Mapping[str, Any],
    jacobian: Sequence[Sequence[float]] | np.ndarray,
    *,
    projection_dim: int,
    seed: int,
    objective: str,
    parameter_pattern: str,
    selected_parameters: int,
    selected_parameter_elements: int,
) -> dict[str, Any]:
    """Return a microbatch row whose local geometry uses policy gradients."""

    matrix = np.asarray(jacobian, dtype=np.float64)
    coefficients = np.asarray(row["estimated_coefficients"], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape != (projection_dim, coefficients.size):
        raise ValueError(
            f"jacobian must have shape ({projection_dim}, {coefficients.size}); got {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("jacobian must contain finite values")
    if np.any(np.linalg.norm(matrix, axis=0) == 0):
        raise ValueError("every response Jacobian column must be non-zero")
    identity = np.eye(projection_dim, dtype=np.float64)
    zero = np.zeros(projection_dim, dtype=np.float64)
    metadata = dict(row.get("metadata", {}))
    metadata.update(
        {
            "geometry_method": "countsketch_policy_logprob_jacobian",
            "reward_geometry_valid": True,
            "jacobian_projection_dim": projection_dim,
            "jacobian_seed": seed,
            "jacobian_objective": objective,
            "jacobian_parameter_pattern": parameter_pattern,
            "jacobian_selected_parameters": selected_parameters,
            "jacobian_selected_parameter_elements": selected_parameter_elements,
            "jacobian_column_norms": np.linalg.norm(matrix, axis=0).tolist(),
        }
    )
    geometry = {"G": matrix.tolist(), "P": identity.tolist(), "h": zero.tolist()}
    return {**row, "geometry": geometry, "reward_geometry": geometry, "metadata": metadata}


def extract_group_policy_jacobian(
    model: Any,
    tokenizer: Any,
    *,
    prompt: Sequence[Mapping[str, str]],
    responses: Sequence[str],
    named_parameters: Sequence[tuple[str, Any]],
    projection_dim: int,
    seed: int,
    objective: str = "mean_logprob",
    max_length: int | None = None,
) -> np.ndarray:
    """Extract one projected response-gradient column per frozen response."""

    import torch
    import torch.nn.functional as F

    if objective not in {"mean_logprob", "sum_logprob"}:
        raise ValueError("objective must be mean_logprob or sum_logprob")
    if len(responses) < 2:
        raise ValueError("a SetGRPO group needs at least two responses")
    parameters = [parameter for _, parameter in named_parameters]
    if not parameters:
        raise ValueError("no parameters selected for Jacobian extraction")
    input_device = _model_input_device(model)
    columns = []
    _prepare_model_for_deterministic_checkpointed_gradients(model)
    for response_index, response in enumerate(responses):
        model.zero_grad(set_to_none=True)
        if input_device.type == "cuda":
            torch.cuda.empty_cache()
        prompt_ids = _prompt_token_ids(tokenizer, prompt)
        response_ids = tokenizer.encode(response, add_special_tokens=False)
        if tokenizer.eos_token_id is not None:
            response_ids = [*response_ids, int(tokenizer.eos_token_id)]
        if not response_ids:
            raise ValueError(f"response {response_index} tokenized to an empty sequence")
        if max_length is not None:
            allowed = max_length - len(prompt_ids)
            if allowed < 1:
                raise ValueError("prompt length leaves no room for response tokens")
            response_ids = response_ids[:allowed]
        token_ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=input_device)
        attention_mask = torch.ones_like(token_ids)
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if input_device.type == "cuda"
            else nullcontext()
        )
        with autocast_context:
            outputs = model(input_ids=token_ids, attention_mask=attention_mask, use_cache=False)
            # Preserve the original FP32 log-probability objective so the
            # memory fix does not change the extracted Jacobian numerics.
            logits = outputs.logits[:, :-1, :].float()
            targets = token_ids[:, 1:]
            token_logprobs = F.log_softmax(logits, dim=-1).gather(
                -1, targets.unsqueeze(-1)
            ).squeeze(-1)
            response_logprobs = token_logprobs[:, max(len(prompt_ids) - 1, 0) :]
            if response_logprobs.numel() == 0:
                raise ValueError(f"response {response_index} has no scored tokens")
            scalar = response_logprobs.mean() if objective == "mean_logprob" else response_logprobs.sum()
        gradients = torch.autograd.grad(
            scalar,
            parameters,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        projected = countsketch_project_named_gradients(
            zip((name for name, _ in named_parameters), gradients),
            projection_dim=projection_dim,
            seed=seed,
        )
        columns.append(projected.detach().cpu().numpy().astype(np.float64, copy=False))
        del outputs, logits, targets, token_logprobs, response_logprobs, scalar, gradients, projected
        del token_ids, attention_mask
        if input_device.type == "cuda":
            torch.cuda.empty_cache()
    return np.stack(columns, axis=1)


def select_jacobian_parameters(model: Any, pattern: str) -> list[tuple[str, Any]]:
    """Select and enable only parameters matching the requested regular expression."""

    expression = re.compile(pattern)
    selected = []
    for name, parameter in model.named_parameters():
        enabled = expression.search(name) is not None
        parameter.requires_grad_(enabled)
        if enabled:
            selected.append((name, parameter))
    if not selected:
        raise ValueError(f"parameter pattern {pattern!r} matched no model parameters")
    return selected


def _enable_gradient_checkpointing(model: Any) -> None:
    """Enable a memory-saving checkpoint mode when supported by Transformers."""

    if not hasattr(model, "gradient_checkpointing_enable"):
        return
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False


def _prepare_model_for_deterministic_checkpointed_gradients(model: Any) -> None:
    """Keep eval semantics while activating checkpoint wrapper layers."""

    model.eval()
    checkpointing_enabled = bool(getattr(model, "is_gradient_checkpointing", False)) or any(
        bool(getattr(module, "gradient_checkpointing", False)) for module in model.modules()
    )
    if not checkpointing_enabled:
        return

    try:
        from transformers.modeling_layers import GradientCheckpointingLayer
    except ImportError:
        GradientCheckpointingLayer = None

    enabled_layers = 0
    if GradientCheckpointingLayer is not None:
        for module in model.modules():
            if isinstance(module, GradientCheckpointingLayer) and bool(
                getattr(module, "gradient_checkpointing", False)
            ):
                module.training = True
                enabled_layers += 1
    if enabled_layers:
        if model.training:
            raise RuntimeError("checkpoint preparation must keep the top-level model in eval mode")
        unexpected_training = [
            name
            for name, module in model.named_modules()
            if module.training and not isinstance(module, GradientCheckpointingLayer)
        ]
        if unexpected_training:
            raise RuntimeError(
                "non-checkpoint modules unexpectedly entered training mode: "
                f"{unexpected_training[:8]}"
            )
        print(f"gradient_checkpointing_eval_layers = {enabled_layers}", flush=True)
        return

    raise RuntimeError(
        "gradient checkpointing is enabled, but no supported "
        "GradientCheckpointingLayer was found; refusing to switch the full "
        "model to train mode because that would change evaluation semantics"
    )


def enrich_rows_with_policy_jacobians(
    rows: Sequence[Mapping[str, Any]],
    transactions: Sequence[Mapping[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    parameter_pattern: str,
    projection_dim: int,
    seed: int,
    objective: str,
    max_length: int | None,
    skip_ids: set[str] | None = None,
    on_enriched: Any | None = None,
) -> tuple[list[dict[str, Any]], JacobianExtractionSummary]:
    """Join transactions to microbatch rows and inject extracted policy geometry."""

    selected = select_jacobian_parameters(model, parameter_pattern)
    transaction_by_root = {
        str(transaction["answer_group"]["query_root"]): transaction for transaction in transactions
    }
    output = []
    skipped = skip_ids or set()
    for index, row in enumerate(rows):
        microbatch_id = str(row["microbatch_id"])
        if microbatch_id in skipped:
            continue
        transaction = transaction_by_root.get(microbatch_id)
        if transaction is None:
            raise ValueError(f"no transaction found for microbatch_id {microbatch_id!r}")
        group = transaction["answer_group"]
        answer_ids = [str(answer["answer_id"]) for answer in group["answers"]]
        expected_ids = list(row.get("metadata", {}).get("answer_ids") or answer_ids)
        if answer_ids != expected_ids:
            raise ValueError(f"answer order mismatch for microbatch {microbatch_id}")
        print(
            f"Extracting policy Jacobian {index + 1}/{len(rows)} for {microbatch_id} "
            f"({len(answer_ids)} responses)",
            flush=True,
        )
        jacobian = extract_group_policy_jacobian(
            model,
            tokenizer,
            prompt=group["prompt"],
            responses=[str(answer["text"]) for answer in group["answers"]],
            named_parameters=selected,
            projection_dim=projection_dim,
            seed=seed,
            objective=objective,
            max_length=max_length,
        )
        enriched = inject_policy_jacobian_geometry(
                row,
                jacobian,
                projection_dim=projection_dim,
                seed=seed,
                objective=objective,
                parameter_pattern=parameter_pattern,
                selected_parameters=len(selected),
                selected_parameter_elements=sum(parameter.numel() for _, parameter in selected),
            )
        output.append(enriched)
        if on_enriched is not None:
            on_enriched(enriched)
    summary = JacobianExtractionSummary(
        input_rows=len(rows),
        enriched_rows=len(skipped) + len(output),
        projection_dim=projection_dim,
        seed=seed,
        objective=objective,
        parameter_pattern=parameter_pattern,
        selected_parameters=len(selected),
        selected_parameter_elements=sum(parameter.numel() for _, parameter in selected),
    )
    return output, summary


def _parameter_seed(name: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _prompt_token_ids(tokenizer: Any, prompt: Sequence[Mapping[str, str]]) -> list[int]:
    if hasattr(tokenizer, "apply_chat_template"):
        ids = tokenizer.apply_chat_template(
            [dict(message) for message in prompt], tokenize=True, add_generation_prompt=True
        )
        return _coerce_token_ids(ids)
    rendered = "\n".join(f"{message['role']}: {message['content']}" for message in prompt) + "\nassistant:"
    return _coerce_token_ids(tokenizer.encode(rendered, add_special_tokens=True))


def _coerce_token_ids(ids: Any) -> list[int]:
    """Normalize tokenizer outputs to a flat list of integer token ids."""

    if isinstance(ids, MappingABC):
        if "input_ids" not in ids:
            raise ValueError(f"tokenizer output mapping lacks input_ids; keys={list(ids.keys())}")
        ids = ids["input_ids"]
    elif hasattr(ids, "input_ids"):
        ids = ids.input_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, tuple):
        ids = list(ids)
    while isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], (list, tuple)):
        ids = list(ids[0])
    if not isinstance(ids, list):
        raise ValueError(f"tokenizer output must be list-like after normalization; got {type(ids).__name__}")
    return [int(value) for value in ids]


def _model_input_device(model: Any) -> Any:
    for parameter in model.parameters():
        if str(parameter.device) != "meta":
            return parameter.device
    raise ValueError("model has no materialized parameter device")


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--rows", required=True, help="Materialized SetGRPO microbatch JSONL")
    root.add_argument("--transactions", required=True, help="Cross-scored transaction JSONL")
    root.add_argument("--output", required=True)
    root.add_argument("--summary")
    root.add_argument("--model", required=True)
    root.add_argument("--adapter")
    root.add_argument("--parameter-pattern", default=r".*norm.*weight$")
    root.add_argument("--projection-dim", type=int, default=256)
    root.add_argument("--seed", type=int, default=2026)
    root.add_argument("--objective", choices=("mean_logprob", "sum_logprob"), default="mean_logprob")
    root.add_argument("--max-length", type=int)
    root.add_argument("--torch-dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    root.add_argument("--device-map", default="auto")
    root.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default=os.getenv("SETGRPO_ATTN_IMPLEMENTATION", "sdpa"),
        help="Attention backend for Jacobian extraction.",
    )
    root.add_argument(
        "--max-memory-gb",
        type=float,
        default=float(os.getenv("SETGRPO_MODEL_MAX_MEMORY_GB", "0")),
        help=(
            "If positive, pass a CUDA max_memory budget to transformers.from_pretrained. "
            "Use this to leave GPU headroom for PEFT adapter loading and gradients."
        ),
    )
    root.add_argument(
        "--adapter-autocast-dtype",
        choices=("0", "1"),
        default=os.getenv("SETGRPO_ADAPTER_AUTOCAST_DTYPE", "0"),
        help=(
            "Whether PEFT should autocast adapter weights, often to fp32. "
            "Default 0 lowers geometry-extraction peak VRAM."
        ),
    )
    root.add_argument(
        "--adapter-low-cpu-mem-usage",
        choices=("0", "1"),
        default=os.getenv("SETGRPO_ADAPTER_LOW_CPU_MEM_USAGE", "0"),
        help="Forward low_cpu_mem_usage to PEFT if supported by the installed version.",
    )
    root.add_argument(
        "--adapter-load-device",
        default=os.getenv("SETGRPO_ADAPTER_LOAD_DEVICE", "auto"),
        help=(
            "Device passed to PEFT load_peft_weights. Use 'cpu' to avoid "
            "loading the full adapter safetensors directly onto CUDA."
        ),
    )
    root.add_argument(
        "--gradient-checkpointing",
        choices=("0", "1"),
        default=os.getenv("SETGRPO_GRADIENT_CHECKPOINTING", "1"),
        help="Enable non-reentrant gradient checkpointing to reduce activation VRAM.",
    )
    root.add_argument("--trust-remote-code", action="store_true")
    root.add_argument("--shard-index", type=int, default=0)
    root.add_argument("--num-shards", type=int, default=1)
    root.add_argument(
        "--resume-rows",
        action="store_true",
        help="Resume from an existing JSONL output by skipping already written microbatch_ids.",
    )
    root.add_argument("--jobs-file", help="JSONL jobs; reuse one loaded model for every job.")
    return root


_MODEL_CACHE: tuple[Any, Any] | None = None


def _run(args: argparse.Namespace) -> None:
    global _MODEL_CACHE
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if _MODEL_CACHE is None:
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[args.torch_dtype]
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        device_map: Any = args.device_map
        if isinstance(device_map, str) and device_map.startswith("cuda:"):
            device_map = {"": int(device_map.split(":", 1)[1])}
        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": device_map,
            "trust_remote_code": args.trust_remote_code,
            "attn_implementation": args.attn_implementation,
            "local_files_only": bool(int(os.getenv("HF_HUB_OFFLINE", "0"))),
        }
        if args.max_memory_gb > 0:
            if not torch.cuda.is_available():
                raise ValueError("--max-memory-gb requires CUDA")
            model_kwargs["max_memory"] = {0: f"{args.max_memory_gb:g}GiB", "cpu": os.getenv("SETGRPO_CPU_MAX_MEMORY", "240GiB")}
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        if args.gradient_checkpointing == "1":
            _enable_gradient_checkpointing(model)
            print("gradient_checkpointing = enabled", flush=True)
        else:
            print("gradient_checkpointing = disabled", flush=True)
        print("model_device_map =", getattr(model, "hf_device_map", None), flush=True)
        print(
            "model_parameter_devices =",
            sorted({str(parameter.device) for parameter in model.parameters()}),
            flush=True,
        )
        _MODEL_CACHE = (model, tokenizer)
    model, tokenizer = _MODEL_CACHE

    rows = _load_jsonl(args.rows)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard-index/num-shards")
    if args.num_shards > 1:
        rows = [row for i, row in enumerate(rows) if i % args.num_shards == args.shard_index]
    transactions = _load_jsonl(args.transactions)
    existing_ids: set[str] = set()
    output_path = Path(args.output)
    if args.resume_rows and output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    existing_ids.add(str(json.loads(line)["microbatch_id"]))
        print(f"Resuming {len(existing_ids)} existing rows from {output_path}", flush=True)

    append_handle = None
    if args.resume_rows:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        append_handle = output_path.open("a", encoding="utf-8")

    def write_checkpoint(row: Mapping[str, Any]) -> None:
        if append_handle is not None:
            append_handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            append_handle.flush()

    enriched, summary = enrich_rows_with_policy_jacobians(
        rows,
        transactions,
        model=model,
        tokenizer=tokenizer,
        parameter_pattern=args.parameter_pattern,
        projection_dim=args.projection_dim,
        seed=args.seed,
        objective=args.objective,
        max_length=args.max_length,
        skip_ids=existing_ids,
        on_enriched=write_checkpoint if append_handle is not None else None,
    )
    if append_handle is not None:
        append_handle.close()
    else:
        _write_jsonl(args.output, enriched)
    summary_path = Path(args.summary) if args.summary else Path(args.output).with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parser().parse_args()
    if args.jobs_file:
        for line in Path(args.jobs_file).read_text(encoding="utf-8").splitlines():
            if line.strip():
                _run(argparse.Namespace(**{**vars(args), **json.loads(line)}))
    else:
        _run(args)


if __name__ == "__main__":
    main()
