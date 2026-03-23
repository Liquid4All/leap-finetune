import logging
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForImageTextToText,
)
from transformers.image_utils import PILImageResampling
from transformers.utils import is_flash_attn_2_available

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_attn_implementation() -> str:
    if is_flash_attn_2_available():
        return "flash_attention_2"
    logger.warning("flash-attn not available, falling back to sdpa")
    return "sdpa"


def _resolve_model_id(model_name: str) -> str:
    """Resolve model_name to a local path or HuggingFace model ID."""
    model_path = Path(model_name)
    if model_path.exists() and model_path.is_dir():
        return model_name
    return f"LiquidAI/{model_name}"


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load only the tokenizer (lightweight, no model weights)."""
    model_id = _resolve_model_id(model_name)
    return AutoTokenizer.from_pretrained(model_id)


def load_model(
    model_name: str,
    model_config: dict | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a model from the Hugging Face Hub or from a local path.

    Args:
        model_name: HuggingFace model ID or local path
        model_config: optional overrides applied to the model config before loading
            (e.g. rope_scaling, max_position_embeddings for YaRN)
    """
    attn_impl = _get_attn_implementation()
    model_id = _resolve_model_id(model_name)
    print(f"Loading model: {model_id}")

    # Load config first so we can apply overrides (rope_scaling, etc.)
    config = AutoConfig.from_pretrained(model_id)
    if model_config:
        for key, value in model_config.items():
            setattr(config, key, value)
        logger.info(f"Applied model config overrides: {list(model_config.keys())}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    # Disable use_cache for training compatibility (gradient checkpointing requires this)
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Use grouped_mm expert routing for MoE (requires torch>=2.9.0)
    if hasattr(model, "set_experts_implementation"):
        model.set_experts_implementation("grouped_mm")

    print(f"Architecture: {model.config.architectures[0]}")
    print(f"Model type: {model.config.model_type}")
    print(f"Layers: {model.config.num_hidden_layers}, Dim: {model.config.hidden_size}")
    print(f"Vocab size: {model.config.vocab_size}")

    return model, tokenizer


def load_vlm_model(
    model_name: str,
    max_image_tokens: int | None = None,
    do_image_splitting: bool = True,
) -> tuple[AutoModelForImageTextToText, AutoProcessor]:
    """Load a VLM model from the Hugging Face Hub or from a local path."""

    processor_kwargs = {
        "trust_remote_code": True,
        "do_image_splitting": do_image_splitting,
        "resample": PILImageResampling.BICUBIC,
    }
    if max_image_tokens is not None:
        processor_kwargs["max_image_tokens"] = max_image_tokens

    model_id = _resolve_model_id(model_name)
    logger.info(f"Loading VLM: {model_id}")

    # SigLIP2 vision encoder doesn't support FA2, use SDPA for the full VLM
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)

    # Disable KV cache for training (required for gradient checkpointing)
    model.config.use_cache = False

    # Ensure padding is configured correctly
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return model, processor
