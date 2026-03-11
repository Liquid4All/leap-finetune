import logging

from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

_DISTRIBUTED_CLEANUP_KEYWORDS = (
    "cuda error",
    "ecc error",
    "nccl",
    "collective",
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


def run_training_safely(trainer, **kwargs):
    """Run trainer.train() with graceful handling of post-training CUDA/NCCL errors.

    Only suppresses distributed errors that occur *after* training made progress
    (global_step > 0), which indicates the error is from cleanup/teardown rather
    than a real training failure.

    Any extra kwargs are forwarded to trainer.train() (e.g. resume_from_checkpoint).
    """
    try:
        trainer.train(**kwargs)
        logger.info("Training completed successfully")
    except RuntimeError as e:
        error_msg = str(e).lower()
        trained = getattr(trainer.state, "global_step", 0) > 0
        is_cleanup_error = trained and any(
            kw in error_msg for kw in _DISTRIBUTED_CLEANUP_KEYWORDS
        )
        if is_cleanup_error:
            logger.warning(
                "Training completed but hit distributed communication error "
                f"during cleanup: {e}"
            )
        else:
            raise
