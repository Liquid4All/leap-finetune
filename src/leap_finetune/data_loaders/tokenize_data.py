import logging

import ray.data
from datasets import Dataset, Features, Sequence, Value
from rich.console import Console
from trl.data_utils import maybe_apply_chat_template, maybe_extract_prompt
from trl.data_utils import pack_dataset

logger = logging.getLogger(__name__)
console = Console()


def tokenize_sft(row: dict, tokenizer, max_length: int) -> dict:
    """
    Tokenize a single SFT row for use in ray_ds.map().

    Handles two formats:
      - Conversational: row has "messages" → apply_chat_template
      - Plain text: row has "text" → tokenizer()
    """
    if "messages" in row:
        result = tokenizer.apply_chat_template(
            row["messages"], tokenize=True, truncation=True, max_length=max_length
        )
        # apply_chat_template returns BatchEncoding (Mapping, not dict)
        input_ids = result["input_ids"] if hasattr(result, "keys") else result
    elif "text" in row:
        input_ids = tokenizer(row["text"], truncation=True, max_length=max_length)[
            "input_ids"
        ]
    else:
        raise ValueError(
            f"Row must have 'messages' or 'text' column, got: {list(row.keys())}"
        )

    return {"input_ids": list(input_ids)}


def tokenize_and_pack_sft(
    ds: ray.data.Dataset,
    tokenizer,
    max_length: int,
    packing: bool = False,
) -> ray.data.Dataset:
    """
    Tokenize and optionally pack an SFT dataset.

    Pipeline:
      1. Distributed tokenization via ray_ds.map()
      2. If packing: materialize to HF Dataset → pack_dataset (BFD) → back to Ray
         If not packing: return directly (tokenizer already truncated)
    """
    # === 1. Distributed tokenization ===
    ds = ds.map(
        tokenize_sft,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
    )

    # === 2. Pack or truncate ===
    if packing:
        # Packing requires full materialization into an HF Dataset
        rows = list(ds.iter_rows())
        features = Features({"input_ids": Sequence(Value("int64"))})
        hf_ds = Dataset.from_list(rows, features=features)
        console.print(f"[dim]Tokenized {len(hf_ds):,} rows[/dim]")
        console.print(f"[dim]Packing sequences (BFD, max_length={max_length})...[/dim]")
        hf_ds = pack_dataset(hf_ds, seq_length=max_length, strategy="bfd")
        console.print(f"[dim]Packed into {len(hf_ds):,} rows[/dim]")
        return ray.data.from_arrow(hf_ds.data.table)

    # Non-packing: tokenizer already truncated to max_length, just return
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

    return {
        "prompt_input_ids": list(prompt_input_ids),
        "chosen_input_ids": list(chosen_input_ids),
        "rejected_input_ids": list(rejected_input_ids),
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
      prompt_input_ids, chosen_input_ids, rejected_input_ids
    """
    ds = ds.map(
        tokenize_dpo,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_prompt_length": max_prompt_length,
            "max_completion_length": max_completion_length,
        },
    )

    # Materialize with proper Arrow types
    rows = list(ds.iter_rows())
    features = Features(
        {
            "prompt_input_ids": Sequence(Value("int64")),
            "chosen_input_ids": Sequence(Value("int64")),
            "rejected_input_ids": Sequence(Value("int64")),
        }
    )
    hf_ds = Dataset.from_list(rows, features=features)
    console.print(f"[dim]Tokenized {len(hf_ds):,} DPO rows[/dim]")

    return ray.data.from_arrow(hf_ds.data.table)
