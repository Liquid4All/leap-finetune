import os
from typing import Union

import ray.train
from peft import get_peft_model
from transformers import AutoTokenizer, PreTrainedModel, ProcessorMixin

from leap_finetune.configs import PeftConfig


def apply_peft_to_model(
    model: PreTrainedModel, peft_config: PeftConfig
) -> PreTrainedModel:
    print("Using PEFT for finetuning")
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    return peft_model


def merge_and_save_peft_model(
    model: PreTrainedModel,
    tokenizer_or_processor: Union[AutoTokenizer, ProcessorMixin],
    output_dir: str,
) -> None:
    """Merge PEFT adapters and save the full model. Only runs on rank 0."""
    # Only rank 0 should save to avoid race conditions and duplicate writes
    if ray.train.get_context().get_world_rank() != 0:
        return

    peft_model_dir = f"{output_dir}/merged_model"
    os.makedirs(peft_model_dir, exist_ok=True)
    print(f"Merging and saving PEFT model to {peft_model_dir}")

    model = model.merge_and_unload()
    model.save_pretrained(peft_model_dir)
    tokenizer_or_processor.save_pretrained(peft_model_dir)
