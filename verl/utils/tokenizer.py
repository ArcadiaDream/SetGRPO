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
"""Utils for tokenization."""

import json
import shutil
import tempfile
import warnings
from pathlib import Path

__all__ = ["hf_tokenizer", "hf_processor"]


_TOKENIZER_COMPAT_DIRS = []


def _auto_tokenizer_from_pretrained_compat(AutoTokenizer, name_or_path, **kwargs):
    """Load tokenizers saved by newer LLaMA-Factory with older Transformers.

    Some SFT checkpoints contain tokenizer_config.json where
    extra_special_tokens is a JSON list. The Transformers version in the verl
    env expects a dict and raises: 'list' object has no attribute 'keys'.  Keep
    the checkpoint untouched by loading from a temporary tokenizer-only overlay
    with extra_special_tokens normalized to {}.
    """
    try:
        return AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    except AttributeError as exc:
        if "'list' object has no attribute 'keys'" not in str(exc):
            raise

    src = Path(str(name_or_path)).expanduser()
    cfg_path = src / "tokenizer_config.json"
    if not src.is_dir() or not cfg_path.is_file():
        raise

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg.get("extra_special_tokens"), list):
        raise
    cfg["extra_special_tokens"] = {}

    tmp = tempfile.TemporaryDirectory(prefix="verl_tokenizer_compat_")
    tmp_path = Path(tmp.name)
    _TOKENIZER_COMPAT_DIRS.append(tmp)

    for item in src.iterdir():
        if item.name == "tokenizer_config.json" or not item.is_file():
            continue
        dst = tmp_path / item.name
        try:
            dst.symlink_to(item)
        except OSError:
            shutil.copy2(item, dst)

    (tmp_path / "tokenizer_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    warnings.warn(
        "Patched tokenizer_config extra_special_tokens list -> dict for older Transformers compatibility: "
        f"{cfg_path}",
        stacklevel=1,
    )
    return AutoTokenizer.from_pretrained(str(tmp_path), **kwargs)


def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}", stacklevel=1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}", stacklevel=1)


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer which correctness handles eos and pad tokens.

    Args:

        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.

    Returns:

        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    from transformers import AutoTokenizer

    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn("Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.", stacklevel=1)
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107
    tokenizer = _auto_tokenizer_from_pretrained_compat(AutoTokenizer, name_or_path, **kwargs)
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def hf_processor(name_or_path, **kwargs):
    """Create a huggingface processor to process multimodal data.

    Args:
        name_or_path (str): The name of the processor.

    Returns:
        transformers.ProcessorMixin: The pretrained processor.
    """
    from transformers import AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except Exception as e:
        processor = None
        # TODO(haibin.lin): try-catch should be removed after adding transformer version req to setup.py to avoid
        # silent failure
        warnings.warn(f"Failed to create processor: {e}. This may affect multimodal processing", stacklevel=1)
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor
