import logging
import os
import pathlib

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

from leap_finetune.utils.loss_utils import install_memory_efficient_causal_lm_loss

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LFM_TIED_WORD_EMBEDDING_MODEL_TYPES = {"lfm2", "lfm2_moe"}
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_LFM2_5_DEFAULT_CHAT_TEMPLATE_PATH = (
    _REPO_ROOT / "job_configs/chat_templates/lfm2_5_chat_template.jinja"
)


def _get_attn_implementation() -> str:
    if is_flash_attn_2_available():
        return "flash_attention_2"
    logger.warning("flash-attn not available, falling back to sdpa")
    return "sdpa"


def _resolve_model_id(model_name: str) -> str:
    """Resolve model_name to a local path or HuggingFace model ID."""
    model_path = pathlib.Path(model_name)
    if model_path.exists() and model_path.is_dir():
        return model_name
    if "/" in model_name or "://" in model_name:
        return model_name
    return f"LiquidAI/{model_name}"


def _is_lfm25_chat_template_model(model_name: str) -> bool:
    lowered = model_name.lower()
    return "lfm2.5" in lowered or "lfm2-24b" in lowered or "lfm2_24b" in lowered


def _default_chat_template_path(model_name: str | None) -> pathlib.Path | None:
    if not model_name:
        return None
    model_path = pathlib.Path(model_name)
    if model_path.exists() and model_path.is_dir():
        return None
    if not _is_lfm25_chat_template_model(model_name):
        return None
    if not _LFM2_5_DEFAULT_CHAT_TEMPLATE_PATH.exists():
        logger.warning(
            "Default LFM2.5 chat template not found at %s",
            _LFM2_5_DEFAULT_CHAT_TEMPLATE_PATH,
        )
        return None
    return _LFM2_5_DEFAULT_CHAT_TEMPLATE_PATH


def _resolve_chat_template(
    chat_template: str | None = None,
    chat_template_path: str | None = None,
    model_name: str | None = None,
) -> str | None:
    if chat_template and chat_template_path:
        raise ValueError("Specify either chat_template or chat_template_path, not both")

    if chat_template_path:
        return pathlib.Path(chat_template_path).expanduser().read_text()

    if chat_template:
        return chat_template

    default_template_path = _default_chat_template_path(model_name)
    if default_template_path:
        logger.info("Using default LFM2.5 chat template: %s", default_template_path)
        return default_template_path.read_text()

    return None


def _apply_chat_template_override(
    tokenizer: AutoTokenizer,
    *,
    chat_template: str | None = None,
    chat_template_path: str | None = None,
    model_name: str | None = None,
) -> AutoTokenizer:
    resolved = _resolve_chat_template(chat_template, chat_template_path, model_name)
    if resolved:
        tokenizer.chat_template = resolved
        logger.info("Applied tokenizer chat template override")
    return tokenizer


def normalize_model_config_overrides(
    config: AutoConfig,
    model_config: dict | None,
) -> dict:
    """Normalize user overrides to the config schema expected by the model."""
    if not model_config:
        return {}

    normalized = dict(model_config)

    # LFM2 / LFM2-MoE consumers do not all read the same RoPE field. Keep the
    # native rope_parameters field populated even when the user only overrides
    # top-level rope_theta.
    if getattr(config, "model_type", "") in LFM_TIED_WORD_EMBEDDING_MODEL_TYPES:
        if normalized.get("tie_word_embeddings") is False:
            raise ValueError("LFM2/LFM2-MoE require tied word embeddings")
        if normalized.get("tie_embedding") is False:
            raise ValueError("LFM2/LFM2-MoE require tied word embeddings")

        rope_parameters = None
        if "rope_parameters" in normalized:
            rope_parameters = dict(normalized["rope_parameters"])
        elif "rope_scaling" in normalized:
            rope_parameters = dict(normalized["rope_scaling"])
        elif "rope_theta" in normalized:
            base_rope_parameters = getattr(config, "rope_parameters", None)
            if isinstance(base_rope_parameters, dict):
                rope_parameters = dict(base_rope_parameters)
            else:
                rope_parameters = {}

        if rope_parameters is not None:
            if normalized.get("rope_theta") is not None:
                rope_parameters["rope_theta"] = normalized["rope_theta"]
            elif (
                "rope_theta" not in rope_parameters
                or rope_parameters["rope_theta"] is None
            ):
                rope_parameters["rope_theta"] = getattr(config, "default_theta", None)
            if "rope_type" not in rope_parameters and "type" in rope_parameters:
                rope_parameters["rope_type"] = rope_parameters["type"]
            normalized["rope_parameters"] = rope_parameters

    return normalized


def _is_moe_model(model: AutoModelForCausalLM) -> bool:
    model_type = getattr(model.config, "model_type", "")
    architectures = getattr(model.config, "architectures", []) or []
    return "moe" in model_type.lower() or any("Moe" in arch for arch in architectures)


def _maybe_enable_grouped_mm(model: AutoModelForCausalLM) -> None:
    """Enable grouped_mm only for MoE models that expose the hook."""
    if os.getenv("LEAP_ENABLE_GROUPED_MM", "1") == "0":
        logger.info("Skipping grouped_mm expert implementation override")
        return
    if not hasattr(model, "set_experts_implementation") or not _is_moe_model(model):
        return
    model.set_experts_implementation("grouped_mm")


def _enforce_lfm_tied_word_embeddings(model: AutoModelForCausalLM) -> None:
    if (
        getattr(model.config, "model_type", "")
        not in LFM_TIED_WORD_EMBEDDING_MODEL_TYPES
    ):
        return
    if not hasattr(model, "tie_weights"):
        return
    model.config.tie_word_embeddings = True
    model.tie_weights()
    logger.info("Enforced tied LFM input/output word embeddings")


def load_tokenizer(
    model_name: str,
    *,
    chat_template: str | None = None,
    chat_template_path: str | None = None,
) -> AutoTokenizer:
    """Load only the tokenizer (lightweight, no model weights)."""
    model_id = _resolve_model_id(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return _apply_chat_template_override(
        tokenizer,
        chat_template=chat_template,
        chat_template_path=chat_template_path,
        model_name=model_name,
    )


def load_model(
    model_name: str,
    model_config: dict | None = None,
    *,
    chat_template: str | None = None,
    chat_template_path: str | None = None,
    install_memory_efficient_loss: bool = True,
    enable_grouped_mm: bool = True,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a model from the Hugging Face Hub or from a local path.

    Args:
        model_name: HuggingFace model ID or local path
        model_config: optional overrides applied to the model config before loading
            (e.g. rope_scaling, max_position_embeddings for YaRN)
    """
    attn_impl = _get_attn_implementation()
    model_id = _resolve_model_id(model_name)
    logger.info(f"Loading model: {model_id}")

    config = AutoConfig.from_pretrained(model_id)
    normalized_model_config = normalize_model_config_overrides(config, model_config)
    if normalized_model_config:
        for key, value in normalized_model_config.items():
            setattr(config, key, value)
        logger.info(
            f"Applied model config overrides: {list(normalized_model_config.keys())}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    _enforce_lfm_tied_word_embeddings(model)
    model.config.use_cache = False
    if (
        install_memory_efficient_loss
        and os.getenv("LEAP_INSTALL_MEMORY_EFFICIENT_LOSS", "1") == "1"
    ):
        install_memory_efficient_causal_lm_loss(model)
    else:
        logger.info("Skipping memory-efficient causal LM loss install")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer = _apply_chat_template_override(
        tokenizer,
        chat_template=chat_template,
        chat_template_path=chat_template_path,
        model_name=model_name,
    )

    if enable_grouped_mm:
        _maybe_enable_grouped_mm(model)
    else:
        logger.info("Skipping grouped_mm expert implementation override")

    logger.info(f"Architecture: {model.config.architectures[0]}")
    logger.info(f"Model type: {model.config.model_type}")
    logger.info(
        f"Layers: {model.config.num_hidden_layers}, Dim: {model.config.hidden_size}"
    )
    logger.info(f"Vocab size: {model.config.vocab_size}")

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
