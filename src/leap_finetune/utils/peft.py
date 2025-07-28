from peft import get_peft_model
from transformers import AutoTokenizer, PreTrainedModel

from leap_finetune.configs import PeftConfig


def apply_peft_to_model(
    model: PreTrainedModel, peft_config: PeftConfig
) -> PreTrainedModel:
    print("Using PEFT for finetuning")
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    return peft_model


def merge_and_save_peft_model(
    model: PreTrainedModel, tokenizer: AutoTokenizer, output_dir: str
) -> None:
    peft_model_dir = f"{output_dir}/merged_model"
    print(f"Merging and saving PEFT model to {peft_model_dir}")

    model = model.merge_and_unload()
    model.save_pretrained(peft_model_dir)
    tokenizer.save_pretrained(peft_model_dir)
