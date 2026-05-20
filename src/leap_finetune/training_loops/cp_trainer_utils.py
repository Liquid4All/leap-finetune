import torch
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach

from leap_finetune.utils.context_parallel import (
    aggregate_cp_loss,
    compute_cp_causal_lm_loss,
    split_batch_for_cp,
    validate_cp_batch_replicated,
)


def cp_enabled(cp_config: dict | None) -> bool:
    return bool(cp_config and cp_config.get("cp_size", 1) > 1)


def split_cp_inputs_if_needed(inputs: dict, cp_config: dict | None) -> dict:
    """Ensure CP loss inputs are local sequence shards.

    Training enters compute_loss after training_step has already split the batch
    and injected global shift_labels. Eval/prediction paths call compute_loss
    directly, so they need this defensive split here.
    """
    if not cp_enabled(cp_config):
        return inputs
    if "labels" not in inputs or "shift_labels" in inputs:
        return inputs
    return split_batch_for_cp(inputs, cp_config["cp_rank"], cp_config["cp_size"])


def scale_cp_loss_for_trainer_num_items(
    trainer: Trainer,
    output,
    *,
    return_outputs: bool,
    num_items_in_batch,
):
    """Mirror Trainer.compute_loss scaling when HF passes num_items_in_batch."""
    if not (
        trainer.args.average_tokens_across_devices and num_items_in_batch is not None
    ):
        return output

    factor = trainer.accelerator.num_processes
    if return_outputs:
        loss, outputs = output
        return loss * factor, outputs
    return output * factor


def compute_cp_loss_for_trainer(
    trainer: Trainer,
    model,
    inputs: dict,
    *,
    return_outputs: bool,
    num_items_in_batch,
):
    inputs = split_cp_inputs_if_needed(inputs, trainer.cp_config)
    output = compute_cp_causal_lm_loss(
        model,
        inputs,
        trainer.cp_config["cp_group"],
        trainer.cp_config["cp_size"],
        return_outputs=return_outputs,
        num_items_in_batch=num_items_in_batch,
    )
    return scale_cp_loss_for_trainer_num_items(
        trainer,
        output,
        return_outputs=return_outputs,
        num_items_in_batch=num_items_in_batch,
    )


def cp_prediction_step(
    trainer: Trainer,
    model,
    inputs,
    prediction_loss_only,
    ignore_keys=None,
):
    if not getattr(trainer, "_cp_eval_batch_validated", False):
        validate_cp_batch_replicated(
            inputs,
            trainer.cp_config["cp_group"],
            trainer.cp_config["cp_rank"],
            trainer.cp_config["cp_size"],
        )
        trainer._cp_eval_batch_validated = True

    inputs = split_batch_for_cp(
        inputs, trainer.cp_config["cp_rank"], trainer.cp_config["cp_size"]
    )
    inputs = trainer._prepare_inputs(inputs)

    if ignore_keys is None:
        if hasattr(trainer.model, "config"):
            ignore_keys = getattr(
                trainer.model.config,
                "keys_to_ignore_at_inference",
                ["past_key_values"],
            )
        else:
            ignore_keys = []

    labels = nested_detach(tuple(inputs.get(name) for name in trainer.label_names))
    if len(labels) == 1:
        labels = labels[0]

    with torch.no_grad():
        with trainer.compute_loss_context_manager():
            loss, outputs = compute_cp_causal_lm_loss(
                model,
                inputs,
                trainer.cp_config["cp_group"],
                trainer.cp_config["cp_size"],
                return_outputs=True,
                num_items_in_batch=None,
            )
        loss = aggregate_cp_loss(
            loss.detach().mean(),
            trainer.cp_config["cp_group"],
            trainer.cp_config["cp_size"],
        )

    if prediction_loss_only:
        return loss, None, None

    if isinstance(outputs, dict):
        logits = tuple(
            value for key, value in outputs.items() if key not in ignore_keys + ["loss"]
        )
    else:
        logits = outputs[1:]

    logits = nested_detach(logits)
    if len(logits) == 1:
        logits = logits[0]
    return loss, logits, labels
