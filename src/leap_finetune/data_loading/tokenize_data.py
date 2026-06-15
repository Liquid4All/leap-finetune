import copy
import logging

import ray
import ray.data
import torch
import pyarrow as pa
import pyarrow.compute as pc
from rich.console import Console
from trl.data_utils import maybe_apply_chat_template, maybe_extract_prompt

from leap_finetune.data_loading.image_loader import load_image
from leap_finetune.data_loading.validate_tool_format import (
    normalize_messages_for_chat_template,
    normalize_row_for_chat_template,
)

logger = logging.getLogger(__name__)
console = Console()
_SFT_PACK_COLUMNS = ("input_ids", "assistant_masks", "completion_mask")


# === VLM Collate ===


def _find_template(seq, template):
    """Yield start indices where template occurs in seq."""
    tlen = len(template)
    for i in range(len(seq) - tlen + 1):
        if seq[i : i + tlen] == template:
            yield i


def create_vlm_collate_fn(processor):
    """Create a collate function with assistant-only label masking.

    Only assistant content + <|im_end|> contribute to loss.
    Images are loaded as PIL and passed to the processor for resize/tiling.
    Bad samples are skipped with a warning instead of crashing the batch.
    """
    tokenizer = processor.tokenizer

    # ChatML: <|im_start|>assistant\n{content}<|im_end|>\n
    response_template_ids = tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def _build_labels(input_ids):
        """Mask everything except assistant content + <|im_end|> with -100."""
        labels = torch.full_like(input_ids, -100)
        batch_size, seq_len = input_ids.shape

        for b in range(batch_size):
            ids = input_ids[b].tolist()
            for tmpl_start in _find_template(ids, response_template_ids):
                content_start = tmpl_start + len(response_template_ids)
                j = content_start
                while j < seq_len and ids[j] != end_token_id:
                    j += 1
                if j >= seq_len:
                    # Truncated turn — no <|im_end|> found, skip to avoid
                    # unmasking garbage (padding / partial next turn)
                    continue
                # Unmask content + <|im_end|>
                content_end = j + 1
                labels[b, content_start:content_end] = input_ids[
                    b, content_start:content_end
                ]

        return labels

    def collate_fn(samples):
        valid_samples = []
        all_loaded_images = []
        skip_count = 0

        for raw in samples:
            # Trainer's dataloader yields {"messages": [...]} dicts from HF Dataset
            conversation = raw["messages"] if isinstance(raw, dict) else raw
            sample_copy = copy.deepcopy(conversation)
            loaded_images = []
            try:
                for message in sample_copy:
                    if message["role"] == "user":
                        for content in message["content"]:
                            if content["type"] == "image" and isinstance(
                                content["image"], str
                            ):
                                img = load_image(content["image"])
                                content["image"] = img
                                loaded_images.append(img)
                valid_samples.append(normalize_messages_for_chat_template(sample_copy))
                all_loaded_images.extend(loaded_images)
            except Exception as e:
                skip_count += 1
                logger.warning(f"Skipping sample in collate: {e}")
                for img in loaded_images:
                    if hasattr(img, "close"):
                        img.close()

        if skip_count > 0:
            logger.info(f"Collate skipped {skip_count}/{len(samples)} samples")

        if len(valid_samples) == 0:
            raise RuntimeError(
                f"Entire batch failed: all {len(samples)} samples had errors"
            )

        try:
            batch = processor.apply_chat_template(
                valid_samples,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            batch["labels"] = _build_labels(batch["input_ids"])
            return batch

        finally:
            for img in all_loaded_images:
                if hasattr(img, "close"):
                    img.close()
            all_loaded_images.clear()

    return collate_fn


# === SFT Tokenization ===


def _final_assistant_span_mask(assistant_masks: list[int]) -> list[int]:
    """Keep only the last contiguous assistant span."""
    if not assistant_masks:
        return assistant_masks

    end = None
    for idx in range(len(assistant_masks) - 1, -1, -1):
        if assistant_masks[idx]:
            end = idx
            break

    if end is None:
        return [0] * len(assistant_masks)

    start = end
    while start > 0 and assistant_masks[start - 1]:
        start -= 1

    output = [0] * len(assistant_masks)
    for idx in range(start, end + 1):
        output[idx] = 1
    return output


def tokenize_sft(
    row: dict,
    tokenizer,
    max_length: int,
    assistant_only_loss: bool = False,
    completion_only_loss: bool = False,
    truncate: bool = True,
) -> dict:
    """
    Tokenize a single SFT row for use in ray_ds.map().

    Handles two formats:
      - Conversational: row has "messages" → apply_chat_template
      - Plain text: row has "text" → tokenizer()
    """
    if "messages" in row:
        need_masks = assistant_only_loss or completion_only_loss
        messages = normalize_messages_for_chat_template(row["messages"])
        result = tokenizer.apply_chat_template(
            messages,
            tools=row.get("tools") or None,
            tokenize=True,
            truncation=truncate,
            max_length=max_length if truncate else None,
            return_dict=need_masks,
            return_assistant_tokens_mask=need_masks,
        )
        # apply_chat_template returns BatchEncoding (Mapping, not dict)
        input_ids = result["input_ids"] if hasattr(result, "keys") else result
    elif "text" in row:
        if assistant_only_loss or completion_only_loss:
            raise ValueError(
                "assistant_only_loss/completion_only_loss require conversational "
                "SFT rows with a 'messages' column"
            )
        input_ids = tokenizer(
            row["text"],
            truncation=truncate,
            max_length=max_length if truncate else None,
        )["input_ids"]
    else:
        raise ValueError(
            f"Row must have 'messages' or 'text' column, got: {list(row.keys())}"
        )

    output = {"input_ids": list(input_ids), "length": len(input_ids)}
    if "messages" in row and (assistant_only_loss or completion_only_loss):
        assistant_masks = list(result["assistant_masks"])
        if len(assistant_masks) != len(output["input_ids"]):
            raise ValueError(
                "assistant mask length mismatch after chat template tokenization"
            )
        if assistant_only_loss:
            output["assistant_masks"] = assistant_masks
        if completion_only_loss:
            output["completion_mask"] = _final_assistant_span_mask(assistant_masks)

    return output


def _pack_sft_arrow_batch(batch: pa.Table, max_length: int) -> pa.Table:
    """Pack one Ray Arrow batch with TRL's BFD algorithm without Python rows."""
    pack_columns = [name for name in _SFT_PACK_COLUMNS if name in batch.column_names]
    if "input_ids" not in pack_columns:
        raise ValueError("SFT packing requires an input_ids column")

    if batch.num_rows == 0:
        arrays = [
            pa.array([], type=batch.schema.field(name).type) for name in pack_columns
        ]
        arrays.append(pa.array([], type=pa.list_(pa.int32())))
        arrays.append(pa.array([], type=pa.int32()))
        return pa.Table.from_arrays(
            arrays, names=pack_columns + ["seq_lengths", "length"]
        )

    # TRL exposes the Arrow packer as a private helper; using it here avoids
    # materializing the full Ray dataset into a Python list on the driver.
    from trl.data_utils import _pack_bfd

    packed = _pack_bfd(
        batch.select(pack_columns),
        seq_length=max_length,
        on_seq_length_overflow="truncate",
    )
    lengths = pc.list_value_length(packed["input_ids"])
    return packed.append_column("length", lengths)


def tokenize_and_pack_sft(
    ds: ray.data.Dataset,
    tokenizer,
    max_length: int,
    packing: bool = False,
    assistant_only_loss: bool = False,
    completion_only_loss: bool = False,
    drop_overlength: bool = False,
) -> ray.data.Dataset:
    """
    Tokenize and optionally pack an SFT dataset.

    Pipeline:
      1. Distributed tokenization via ray_ds.map()
      2. If packing: pack each Arrow batch with BFD in Ray
         If not packing: return directly (tokenizer already truncated)
    """
    # === 1. Distributed tokenization ===
    ds = ds.map(
        tokenize_sft,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": max_length,
            "assistant_only_loss": assistant_only_loss,
            "completion_only_loss": completion_only_loss,
            "truncate": not drop_overlength,
        },
    )

    if drop_overlength:
        # For long-context SFT we often prefilter complete conversations. Do not
        # silently turn a complete example into a partial supervised target if
        # the active tokenizer/template now renders it over the configured limit.
        ds = ds.filter(lambda row: row["length"] <= max_length)

    # === 2. Pack or truncate ===
    if packing:
        console.print(f"[dim]Packing sequences (BFD, max_length={max_length})...[/dim]")
        return ds.map_batches(
            _pack_sft_arrow_batch,
            batch_format="pyarrow",
            fn_kwargs={"max_length": max_length},
        )

    # Non-packing: tokenizer already truncated to max_length or overlength rows
    # were explicitly dropped above.
    return ds


# === DPO Tokenization ===


def tokenize_dpo(
    row: dict,
    tokenizer,
    max_prompt_length: int | None,
    max_completion_length: int | None,
) -> dict:
    """
    Tokenize a single DPO row for use in ray_ds.map().

    Replicates DPOTrainer's pipeline:
      1. maybe_extract_prompt — extract shared prompt from chosen/rejected
      2. maybe_apply_chat_template — convert messages → strings
      3. tokenize_row — tokenize + truncate + append eos

    Produces: prompt_input_ids, chosen_input_ids, rejected_input_ids
    """
    row = normalize_row_for_chat_template(row)

    # Extract prompt if not already present
    row = maybe_extract_prompt(row)

    # Apply chat template (converts conversational → strings, no-op for strings)
    row = maybe_apply_chat_template(row, tokenizer)

    # Tokenize the 3 string sequences
    prompt_input_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
    chosen_input_ids = tokenizer(row["chosen"], add_special_tokens=False)["input_ids"]
    rejected_input_ids = tokenizer(row["rejected"], add_special_tokens=False)[
        "input_ids"
    ]

    # Append eos to completions (matches DPOTrainer.tokenize_row)
    chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
    rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

    # Truncate: prompt from the left, completions from the right
    if max_prompt_length is not None:
        prompt_input_ids = prompt_input_ids[-max_prompt_length:]
    if max_completion_length is not None:
        chosen_input_ids = chosen_input_ids[:max_completion_length]
        rejected_input_ids = rejected_input_ids[:max_completion_length]

    # Column names must match TRL v1's DPO data collator:
    # prompt_ids, chosen_ids, rejected_ids (changed from *_input_ids in TRL 0.x)
    return {
        "prompt_ids": list(prompt_input_ids),
        "chosen_ids": list(chosen_input_ids),
        "rejected_ids": list(rejected_input_ids),
    }


def tokenize_dpo_dataset(
    ds: ray.data.Dataset,
    tokenizer,
    max_prompt_length: int | None = None,
    max_completion_length: int | None = None,
) -> ray.data.Dataset:
    """
    Tokenize a DPO dataset via Ray .map().

    Returns a Ray Dataset with columns:
      prompt_ids, chosen_ids, rejected_ids (TRL v1 column names)
    """
    ds = ds.map(
        tokenize_dpo,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_prompt_length": max_prompt_length,
            "max_completion_length": max_completion_length,
        },
    )
    return ds
