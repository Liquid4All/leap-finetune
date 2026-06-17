from types import SimpleNamespace

from torch.utils.data import Dataset

from leap_finetune.data_loaders.sampling import get_length_grouped_sampler
from leap_finetune.utils.trainer_mixins import RayDataLoaderMixin


class _TinyDataset(Dataset):
    column_names = ["input_ids", "length"]

    def __init__(self):
        self.rows = [
            {"input_ids": [0], "length": 1},
            {"input_ids": [1, 1], "length": 2},
            {"input_ids": [2, 2, 2], "length": 3},
        ]

    def __getitem__(self, index):
        if index == "length":
            return [row["length"] for row in self.rows]
        return self.rows[index]

    def __len__(self):
        return len(self.rows)


class _Trainer(RayDataLoaderMixin):
    def __init__(self):
        self.args = SimpleNamespace(per_device_train_batch_size=1)
        self.train_dataset = _TinyDataset()
        self.data_collator = lambda rows: rows
        self.cp_config = None


def test_length_grouped_sampler_can_be_disabled():
    dataset = _TinyDataset()

    assert get_length_grouped_sampler(dataset, batch_size=1) is not None
    assert get_length_grouped_sampler(dataset, batch_size=1, enabled=False) is None


def test_ray_dataloader_can_preserve_dataset_order():
    trainer = _Trainer()
    trainer.train_dataloader_shuffle = False

    dataloader = trainer.get_train_dataloader()
    observed = [batch[0]["input_ids"][0] for batch in dataloader]

    assert observed == [0, 1, 2]
