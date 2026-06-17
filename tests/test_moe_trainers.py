from torch.utils.data import Dataset

from leap_finetune.training.moe_sft import LFMMoeSFTTrainer


class LengthDataset(Dataset):
    column_names = ["length"]

    def __init__(self, lengths: list[int]) -> None:
        self.lengths = lengths

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, item):
        if isinstance(item, str):
            if item != "length":
                raise KeyError(item)
            return self.lengths
        return {"length": self.lengths[item]}


def test_moe_sft_length_grouping_uses_dp_seed_for_ep():
    trainer = LFMMoeSFTTrainer.__new__(LFMMoeSFTTrainer)
    trainer.train_dataset = LengthDataset([8, 3, 5, 2])
    trainer._train_batch_size = 2
    trainer.data_collator = lambda features: features
    trainer.ep_config = {"dp_rank": 3}

    dataloader = trainer.get_train_dataloader()

    assert dataloader.sampler is not None
    assert dataloader.sampler.generator is not None
    assert dataloader.sampler.generator.initial_seed() == 45
