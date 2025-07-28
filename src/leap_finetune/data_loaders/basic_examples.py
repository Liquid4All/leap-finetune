from datasets import Dataset, load_dataset
from leap_finetune.data_loaders.validate_loader import validate_data_loader


@validate_data_loader
def load_smoltalk_dataset(limit: int = 10_000) -> tuple[Dataset, Dataset]:
    dataset = load_dataset("HuggingFaceTB/smoltalk", "all", split=f"train[:{limit}]")
    dataset = dataset.train_test_split(test_size=0.2)
    return dataset["train"], dataset["test"]


@validate_data_loader
def load_orpo_dpo_dataset(limit: int = 10_000) -> tuple[Dataset, Dataset]:
    dataset = load_dataset("mlabonne/orpo-dpo-mix-40k", split=f"train[:{limit}]")
    dataset = dataset.train_test_split(test_size=0.2)
    return dataset["train"], dataset["test"]
