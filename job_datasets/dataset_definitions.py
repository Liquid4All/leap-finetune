from leap_finetune.data_loaders.dataset_loader import DatasetLoader

# Example of pre-defining a dataset loader
custom_dataset = DatasetLoader(
    "HuggingFaceTB/smoltalk", "sft", limit=1000, test_size=0.2, subset="all"
)
