import torch
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LengthGroupedSampler


def get_length_grouped_sampler(
    dataset: Dataset | None,
    batch_size: int,
    *,
    generator: torch.Generator | None = None,
) -> LengthGroupedSampler | None:
    """Build a local length-grouped sampler when the dataset exposes a length column."""
    if dataset is None:
        return None

    column_names = getattr(dataset, "column_names", None)
    if column_names is None or "length" not in column_names:
        return None

    lengths = list(dataset["length"])
    if not lengths:
        return None

    return LengthGroupedSampler(
        batch_size=batch_size,
        lengths=lengths,
        generator=generator,
    )
