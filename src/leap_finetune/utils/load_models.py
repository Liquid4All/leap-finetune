from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForImageTextToText,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a model from the Hugging Face Hub or from a local path"""

    # Check if model_name is a local path
    model_path = Path(model_name)
    if model_path.exists() and model_path.is_dir():
        # Load from local path (for checkpoints)
        print(f"Loading model from local path: {model_name}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)

    else:
        # Load from Hugging Face
        model_id = f"LiquidAI/{model_name}"
        print(f"Loading model from Hub: {model_id}")

        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Architecture: {model.config.architectures[0]}")
    print(f"Model type: {model.config.model_type}")
    print(f"Layers: {model.config.num_hidden_layers}, Dim: {model.config.hidden_size}")
    print(f"Vocab size: {model.config.vocab_size}")

    return model, tokenizer


def load_vlm_model(
    model_name: str,
) -> tuple[AutoModelForImageTextToText, AutoProcessor]:
    """Load a VLM model from the Hugging Face Hub or from a local path"""

    # Check if model_name is a local path
    model_path = Path(model_name)
    if model_path.exists() and model_path.is_dir():
        # Load from local path (for checkpoints)
        print(f"Loading model from local path: {model_name}")

        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            max_image_tokens=256,
        )

    else:
        # Load from Hugging Face
        model_id = f"LiquidAI/{model_name}"
        print(f"Loading model from Hub: {model_id}")

        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            max_image_tokens=256,
        )

    return model, processor
