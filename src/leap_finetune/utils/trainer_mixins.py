import logging

from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

_DISTRIBUTED_ERROR_KEYWORDS = (
    "cuda error",
    "ecc error",
    "nccl",
    "collective",
    "timeout",
)


class RayDataLoaderMixin:
    """Bypasses Accelerate's DistributedSampler for Ray-sharded data.

    Handles overriding Accelerate's DistributedSampler for Ray-sharded data.

    Note: prepare_trainer() is a no-op for our setup — it only activates for
    _IterableFromIterator datasets (Ray streaming), not materialized HF Datasets.
    """

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
        )


def run_training_safely(trainer):
    """Run trainer.train() with graceful handling of post-training CUDA/NCCL errors."""
    try:
        trainer.train()
        logger.info("Training completed successfully")
    except RuntimeError as e:
        error_msg = str(e)
        if any(kw in error_msg.lower() for kw in _DISTRIBUTED_ERROR_KEYWORDS):
            logger.warning(
                "Training completed but hit distributed communication error "
                f"during cleanup: {error_msg}"
            )
        else:
            raise
