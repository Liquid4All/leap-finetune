from leap_finetune.configs import PeftConfig, TrainingConfig
from leap_finetune.configs.job_config import JobConfig
from leap_finetune.data_loaders.dataset_loader import DatasetLoader


#################################
#         Dataset Loader       #
#################################

"""
Load your datasets here using the custom DatasetLoader class.

Args:
    dataset_path: Dataset identifier from HuggingFace Hub or local dataset path in JSON/JSONL format
    dataset_type: Type of dataset for training
    limit: Maximum number of samples to load, default None for all samples
    split: Dataset split to use, default "train"
    test_size: Fraction of data to use for testing, default 0.2
    subset: Dataset subset to use for HuggingFace datasets, default None
"""

example_sft_dataset = DatasetLoader(
    "HuggingFaceTB/smoltalk", "sft", limit=1000, test_size=0.2, subset="all"
)

example_dpo_dataset = DatasetLoader(
    "mlabonne/orpo-dpo-mix-40k", "dpo", limit=1000, test_size=0.2, subset="default"
)


#################################
#       Training Config         #
#################################

"""
Choose a default training config and override some of the parameters.

Default SFT: TrainingConfig.DEFAULT_SFT
Default DPO: TrainingConfig.DEFAULT_DPO

Default LORA: PeftConfig.DEFAULT_LORA
High R LORA: PeftConfig.HIGH_R_LORA
No LORA (full finetuning): PeftConfig.NO_LORA

Args:
    output_dir: Output directory for training artifacts
    num_train_epochs: Number of training epochs
    per_device_train_batch_size: Batch size per device
    learning_rate: Learning rate
"""

user_config = {
    "output_dir": None,
    "num_train_epochs": None,
    "per_device_train_batch_size": None,
    "learning_rate": None,
}
training_config = TrainingConfig.DEFAULT_SFT.override(**user_config)
peft_config = PeftConfig.DEFAULT_LORA


#################################
#      Job Configuration        #
#################################

"""
Set up your training job here using JobConfig.

Args:
    job_name: Unique identifier for the training job (alphanumeric, hyphens, underscores only)
    model_name: Model to fine-tune - default: "LFM2-1.2B". Try also "LFM2-700M" or "LFM2-350M"
    training_type: "sft" or "dpo"
    dataset: Dataset to use for training, defined in step 1
    training_config: Training configuration that matches training_type, defined in step 2
    peft_config: PEFT configuration for parameter-efficient fine-tuning, defined in step 2
"""

JOB_CONFIG = JobConfig(
    job_name="my_job",
    model_name="LFM2-1.2B",
    training_type="sft",
    dataset=example_sft_dataset,
    training_config=training_config,
    peft_config=peft_config,
)
