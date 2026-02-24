import pathlib
import re
from datetime import datetime

from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, PreTrainedModel, ProcessorMixin

from leap_finetune.training_configs import PeftConfig


def apply_peft_to_model(
    model: PreTrainedModel, peft_config: PeftConfig | LoraConfig
) -> PreTrainedModel:
    print("Using PEFT for finetuning")

    # Handle enum, _CustomPeftConfig, and direct LoraConfig
    if hasattr(peft_config, "value"):
        config = peft_config.value
    else:
        config = peft_config

    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()

    return peft_model


def merge_and_save_peft_model(
    model: PreTrainedModel,
    tokenizer_or_processor: AutoTokenizer | ProcessorMixin,
    output_dir: str,
    run_name_template: str | None = None,
) -> None:
    output_path = pathlib.Path(output_dir)

    if run_name_template:
        merged_name = run_name_template.replace("-lora_a-", "-lora_m-")
        # Strip epoch/step suffix if present (e.g. -e1s100 or -e1)
        merged_name = re.sub(r"-e\d+(s\d+)?$", "", merged_name)
    else:
        # Fallback: scan existing checkpoint dirs for a name to derive from
        merged_name = None
        if output_path.exists():
            for child in output_path.iterdir():
                if child.is_dir() and "-lora_a-" in child.name:
                    base = re.sub(r"-e\d+.*", "", child.name)
                    merged_name = base.replace("-lora_a-", "-lora_m-")
                    break

        if not merged_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            merged_name = f"merged_model-{timestamp}"

    peft_model_dir = output_path / merged_name
    peft_model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Merging and saving PEFT model to {peft_model_dir}")

    model = model.merge_and_unload()
    model.save_pretrained(str(peft_model_dir))
    tokenizer_or_processor.save_pretrained(str(peft_model_dir))
