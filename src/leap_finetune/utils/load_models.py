import logging
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForImageTextToText,
)
from transformers.image_utils import PILImageResampling

logger = logging.getLogger(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a model from the Hugging Face Hub or from a local path"""

    # Check if model_name is a local path
    model_path = Path(model_name)
    if model_path.exists() and model_path.is_dir():
        # Load from local path (for checkpoints)
        print(f"Loading model from local path: {model_name}")

        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
        # Disable use_cache for training compatibility (gradient checkpointing requires this)
        model.config.use_cache = False

        tokenizer = AutoTokenizer.from_pretrained(model_name)

    else:
        # Load from Hugging Face
        model_id = f"LiquidAI/{model_name}"
        print(f"Loading model from Hub: {model_id}")

        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
        # Disable use_cache for training compatibility (gradient checkpointing requires this)
        model.config.use_cache = False

        tokenizer = AutoTokenizer.from_pretrained(model_id)

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

    # Check if model_name is a local path
    model_path = Path(model_name)
    if model_path.exists() and model_path.is_dir():
        logger.info(f"Loading VLM from local path: {model_name}")

        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)

    else:
        model_id = f"LiquidAI/{model_name}"
        logger.info(f"Loading VLM from Hub: {model_id}")

        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)

    # Disable KV cache for training (required for gradient checkpointing)
    model.config.use_cache = False

    # Ensure padding is configured correctly
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return model, processor
