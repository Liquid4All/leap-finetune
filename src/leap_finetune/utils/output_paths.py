from leap_finetune.utils.constants import SFT_OUTPUT_PATH, DPO_OUTPUT_PATH
from pathlib import Path


def resolve_model_output_path(training_type: str, job_name: str) -> Path:
    """
    Resolves the model output and ray storage paths for a given training job.
    """
    base_path = SFT_OUTPUT_PATH if training_type == "sft" else DPO_OUTPUT_PATH
    model_output_dir = base_path / job_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    return model_output_dir


def is_job_name_unique(training_type: str, job_name: str) -> bool:
    """
    Checks if a job with the given name exists for a given training type.
    """
    base_path = SFT_OUTPUT_PATH if training_type == "sft" else DPO_OUTPUT_PATH
    model_output_dir = base_path / job_name
    return not model_output_dir.exists()
