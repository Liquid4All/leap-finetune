#!/usr/bin/env python3
"""Build an action-anchored curriculum dataset for Shopify Suggested Actions.

The output keeps the normal conversational schema (`messages` + `tools`) and
adds audit metadata. It does not bake chat-template text into the dataset.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

CREATE_ACTION = "create_action"
DEFAULT_PRESERVE_TOOLS = (
    "fetch_store_analytics",
    "shopify_admin_graphql_agent",
    "query_customer_segment",
    "web_search_agent",
    "search_shop_resource",
    "fetch_help_documents",
)
BASE_COLUMNS = ("shop_id", "messages", "tools", "conversation_id")
METADATA_FIELDS = (
    pa.field("source_split", pa.string()),
    pa.field("source_row_index", pa.int64()),
    pa.field("source_file", pa.string()),
    pa.field("source_file_row_index", pa.int64()),
    pa.field("source_message_count", pa.int32()),
    pa.field("prefix_message_count", pa.int32()),
    pa.field("target_type", pa.string()),
    pa.field("target_message_index", pa.int32()),
    pa.field("target_assistant_ordinal", pa.int32()),
    pa.field("target_create_action_ordinal", pa.int32()),
    pa.field("target_tool_names", pa.list_(pa.string())),
    pa.field("curriculum_phase", pa.int32()),
    pa.field("curriculum_rank", pa.int64()),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="/lambdafs/alay/datasets/Shopify_sidekick-suggested-actions-distillation",
        help="Local Shopify dataset mirror containing data/*.parquet.",
    )
    parser.add_argument(
        "--output",
        default="/lambdafs/alay/datasets/shopify-suggested-actions-action-curriculum",
        help="Output directory for train/ and validation/ parquet shards.",
    )
    parser.add_argument(
        "--source-rows-per-shard",
        type=int,
        default=256,
        help="Number of source conversations per output shard before phase ordering.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Parquet read batch size.",
    )
    parser.add_argument(
        "--final-summary-fraction",
        type=float,
        default=0.25,
        help="Fraction of conversations with full final-summary rows included.",
    )
    parser.add_argument(
        "--research-preserve-fraction",
        type=float,
        default=0.15,
        help="Fraction of conversations with one non-create tool-call prefix included.",
    )
    parser.add_argument(
        "--preserve-tools",
        nargs="*",
        default=list(DEFAULT_PRESERVE_TOOLS),
        help="Preferred non-create tool names for the preservation stream.",
    )
    parser.add_argument(
        "--include-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write full validation rows for held-out evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic sampling seed.",
    )
    parser.add_argument(
        "--max-source-rows",
        type=int,
        default=None,
        help="Optional limit for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output directory.",
    )
    return parser.parse_args()


def list_split_files(input_dir: Path, split: str) -> list[Path]:
    files = sorted((input_dir / "data").glob(f"{split}-*.parquet"))
    if not files:
        files = sorted(input_dir.glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No {split} parquet files found under {input_dir}")
    return files


def build_output_schema(source_schema: pa.Schema) -> pa.Schema:
    fields = [source_schema.field(name) for name in BASE_COLUMNS]
    return pa.schema([*fields, *METADATA_FIELDS])


def tool_call_names(message: dict[str, Any]) -> list[str]:
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list):
        return []

    names = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            function = call
        name = function.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def assistant_ordinal(messages: list[dict[str, Any]], target_index: int) -> int:
    return sum(
        1
        for message in messages[: target_index + 1]
        if message.get("role") == "assistant"
    )


def create_action_ordinal(
    create_indices: list[int],
    target_index: int,
) -> int:
    for ordinal, index in enumerate(create_indices, start=1):
        if index == target_index:
            return ordinal
    return 0


def make_prefix_row(
    row: dict[str, Any],
    *,
    source_split: str,
    source_row_index: int,
    source_file: Path,
    source_file_row_index: int,
    target_type: str,
    target_index: int,
    phase: int,
    rank: int,
    create_indices: list[int],
) -> dict[str, Any]:
    messages = row["messages"]
    target_names = tool_call_names(messages[target_index])
    return {
        "shop_id": row.get("shop_id"),
        "messages": messages[: target_index + 1],
        "tools": row.get("tools"),
        "conversation_id": row.get("conversation_id"),
        "source_split": source_split,
        "source_row_index": source_row_index,
        "source_file": source_file.name,
        "source_file_row_index": source_file_row_index,
        "source_message_count": len(messages),
        "prefix_message_count": target_index + 1,
        "target_type": target_type,
        "target_message_index": target_index,
        "target_assistant_ordinal": assistant_ordinal(messages, target_index),
        "target_create_action_ordinal": create_action_ordinal(
            create_indices, target_index
        ),
        "target_tool_names": target_names,
        "curriculum_phase": phase,
        "curriculum_rank": rank,
    }


def select_research_preserve_index(
    messages: list[dict[str, Any]],
    *,
    rng: random.Random,
    preserve_tools: set[str],
) -> int | None:
    candidates: list[tuple[int, list[str]]] = []
    preferred: list[tuple[int, list[str]]] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        names = tool_call_names(message)
        if not names or CREATE_ACTION in names:
            continue
        item = (index, names)
        candidates.append(item)
        if preserve_tools.intersection(names):
            preferred.append(item)

    pool = preferred or candidates
    if not pool:
        return None
    return rng.choice(pool)[0]


def selected_create_indices(create_indices: list[int]) -> list[tuple[str, int, int]]:
    if not create_indices:
        return []

    selections = [
        ("first_create_action", create_indices[0], 0),
        ("middle_create_action", create_indices[len(create_indices) // 2], 2),
        ("last_create_action", create_indices[-1], 3),
    ]
    deduped = []
    seen = set()
    for target_type, index, phase in selections:
        if index in seen:
            continue
        deduped.append((target_type, index, phase))
        seen.add(index)
    return deduped


def build_train_candidates(
    row: dict[str, Any],
    *,
    source_split: str,
    source_row_index: int,
    source_file: Path,
    source_file_row_index: int,
    rng: random.Random,
    preserve_tools: set[str],
    final_summary_fraction: float,
    research_preserve_fraction: float,
    rank_start: int,
) -> list[dict[str, Any]]:
    messages = row.get("messages") or []
    if not isinstance(messages, list):
        return []

    create_indices = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "assistant"
        and CREATE_ACTION in tool_call_names(message)
    ]
    if not create_indices:
        return []

    rows = []
    rank = rank_start
    for target_type, target_index, phase in selected_create_indices(create_indices):
        rows.append(
            make_prefix_row(
                row,
                source_split=source_split,
                source_row_index=source_row_index,
                source_file=source_file,
                source_file_row_index=source_file_row_index,
                target_type=target_type,
                target_index=target_index,
                phase=phase,
                rank=rank,
                create_indices=create_indices,
            )
        )
        rank += 1

    if rng.random() < research_preserve_fraction:
        preserve_index = select_research_preserve_index(
            messages,
            rng=rng,
            preserve_tools=preserve_tools,
        )
        if preserve_index is not None:
            rows.append(
                make_prefix_row(
                    row,
                    source_split=source_split,
                    source_row_index=source_row_index,
                    source_file=source_file,
                    source_file_row_index=source_file_row_index,
                    target_type="research_tool_preserve",
                    target_index=preserve_index,
                    phase=1,
                    rank=rank,
                    create_indices=create_indices,
                )
            )
            rank += 1

    final_index = len(messages) - 1
    if (
        final_index >= 0
        and messages[final_index].get("role") == "assistant"
        and rng.random() < final_summary_fraction
    ):
        rows.append(
            make_prefix_row(
                row,
                source_split=source_split,
                source_row_index=source_row_index,
                source_file=source_file,
                source_file_row_index=source_file_row_index,
                target_type="final_summary",
                target_index=final_index,
                phase=4,
                rank=rank,
                create_indices=create_indices,
            )
        )

    return rows


def write_rows(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    arrays = []
    for field in schema:
        arrays.append(pa.array([row.get(field.name) for row in rows], type=field.type))
    table = pa.Table.from_arrays(arrays, schema=schema)
    pq.write_table(table, path, compression="zstd")


def flush_train_shard(
    output_dir: Path,
    shard_index: int,
    phase_rows: dict[int, list[dict[str, Any]]],
    schema: pa.Schema,
) -> int:
    rows = []
    for phase in sorted(phase_rows):
        rows.extend(phase_rows[phase])
    if not rows:
        return shard_index

    path = output_dir / f"train-{shard_index:05d}.parquet"
    write_rows(path, rows, schema)
    return shard_index + 1


def iter_parquet_rows(files: list[Path], batch_size: int):
    for file in files:
        file_row_index = 0
        parquet_file = pq.ParquetFile(file)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                yield file, file_row_index, row
                file_row_index += 1


def build_train_split(
    *,
    input_dir: Path,
    output_dir: Path,
    schema: pa.Schema,
    args: argparse.Namespace,
) -> dict[str, Any]:
    train_files = list_split_files(input_dir, "train")
    rng = random.Random(args.seed)
    preserve_tools = set(args.preserve_tools)
    phase_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    source_rows_in_shard = 0
    shard_index = 0
    source_row_index = 0
    curriculum_rank = 0
    stats: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()

    for source_file, file_row_index, row in iter_parquet_rows(
        train_files, args.batch_size
    ):
        if (
            args.max_source_rows is not None
            and source_row_index >= args.max_source_rows
        ):
            break

        candidates = build_train_candidates(
            row,
            source_split="train",
            source_row_index=source_row_index,
            source_file=source_file,
            source_file_row_index=file_row_index,
            rng=rng,
            preserve_tools=preserve_tools,
            final_summary_fraction=args.final_summary_fraction,
            research_preserve_fraction=args.research_preserve_fraction,
            rank_start=curriculum_rank,
        )
        curriculum_rank += len(candidates)

        if candidates:
            stats["source_rows_with_candidates"] += 1
            for candidate in candidates:
                phase_rows[candidate["curriculum_phase"]].append(candidate)
                target_counts[candidate["target_type"]] += 1
                tool_counts.update(candidate["target_tool_names"])
        else:
            stats["source_rows_without_candidates"] += 1

        source_row_index += 1
        source_rows_in_shard += 1
        if source_rows_in_shard >= args.source_rows_per_shard:
            shard_index = flush_train_shard(output_dir, shard_index, phase_rows, schema)
            phase_rows = defaultdict(list)
            source_rows_in_shard = 0

    shard_index = flush_train_shard(output_dir, shard_index, phase_rows, schema)
    stats["source_rows_seen"] = source_row_index
    stats["output_rows"] = sum(target_counts.values())
    stats["output_shards"] = shard_index
    return {
        "stats": dict(stats),
        "target_counts": dict(target_counts),
        "target_tool_counts": dict(tool_counts),
    }


def build_validation_split(
    *,
    input_dir: Path,
    output_dir: Path,
    schema: pa.Schema,
    args: argparse.Namespace,
) -> dict[str, Any]:
    validation_files = list_split_files(input_dir, "validation")
    shard_rows = []
    shard_index = 0
    source_row_index = 0
    stats: Counter[str] = Counter()

    for source_file, file_row_index, row in iter_parquet_rows(
        validation_files, args.batch_size
    ):
        if (
            args.max_source_rows is not None
            and source_row_index >= args.max_source_rows
        ):
            break

        messages = row.get("messages") or []
        create_indices = [
            index
            for index, message in enumerate(messages)
            if isinstance(message, dict)
            and message.get("role") == "assistant"
            and CREATE_ACTION in tool_call_names(message)
        ]
        target_index = len(messages) - 1 if messages else -1
        output_row = {
            "shop_id": row.get("shop_id"),
            "messages": messages,
            "tools": row.get("tools"),
            "conversation_id": row.get("conversation_id"),
            "source_split": "validation",
            "source_row_index": source_row_index,
            "source_file": source_file.name,
            "source_file_row_index": file_row_index,
            "source_message_count": len(messages),
            "prefix_message_count": len(messages),
            "target_type": "full_validation",
            "target_message_index": target_index,
            "target_assistant_ordinal": assistant_ordinal(messages, target_index)
            if target_index >= 0
            else 0,
            "target_create_action_ordinal": 0,
            "target_tool_names": tool_call_names(messages[target_index])
            if target_index >= 0
            else [],
            "curriculum_phase": 0,
            "curriculum_rank": source_row_index,
        }
        if create_indices:
            stats["source_rows_with_create_action"] += 1
        shard_rows.append(output_row)
        source_row_index += 1

        if len(shard_rows) >= args.source_rows_per_shard:
            path = output_dir / f"validation-{shard_index:05d}.parquet"
            write_rows(path, shard_rows, schema)
            shard_rows = []
            shard_index += 1

    if shard_rows:
        path = output_dir / f"validation-{shard_index:05d}.parquet"
        write_rows(path, shard_rows, schema)
        shard_index += 1

    stats["source_rows_seen"] = source_row_index
    stats["output_rows"] = source_row_index
    stats["output_shards"] = shard_index
    return {"stats": dict(stats)}


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if args.source_rows_per_shard <= 0:
        raise ValueError("--source-rows-per-shard must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    for name in ("final_summary_fraction", "research_preserve_fraction"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_dir} already exists; pass --overwrite to replace it"
            )
        shutil.rmtree(output_dir)
    train_output_dir = output_dir / "train"
    validation_output_dir = output_dir / "validation"
    train_output_dir.mkdir(parents=True)
    if args.include_validation:
        validation_output_dir.mkdir(parents=True)

    first_train_file = list_split_files(input_dir, "train")[0]
    source_schema = pq.ParquetFile(first_train_file).schema_arrow
    output_schema = build_output_schema(source_schema)

    train_summary = build_train_split(
        input_dir=input_dir,
        output_dir=train_output_dir,
        schema=output_schema,
        args=args,
    )
    validation_summary = None
    if args.include_validation:
        validation_summary = build_validation_split(
            input_dir=input_dir,
            output_dir=validation_output_dir,
            schema=output_schema,
            args=args,
        )

    summary = {
        "input": str(input_dir),
        "output": str(output_dir),
        "seed": args.seed,
        "source_rows_per_shard": args.source_rows_per_shard,
        "final_summary_fraction": args.final_summary_fraction,
        "research_preserve_fraction": args.research_preserve_fraction,
        "preserve_tools": args.preserve_tools,
        "train": train_summary,
        "validation": validation_summary,
        "recommended_training_config": {
            "assistant_only_loss": True,
            "completion_only_loss": False,
            "shuffle_dataset": False,
            "train_dataloader_shuffle": False,
            "length_grouped_sampling": False,
        },
    }
    (output_dir / "curriculum_stats.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "README.md").write_text(
        "# Shopify Suggested Actions Action Curriculum\n\n"
        "Action-anchored prefixes generated from the raw Shopify Suggested "
        "Actions trajectories. Rows keep `messages` and `tools` structured; "
        "chat-template rendering is intentionally deferred to training.\n\n"
        "Recommended loss/order settings:\n\n"
        "```yaml\n"
        "assistant_only_loss: true\n"
        "completion_only_loss: false\n"
        "shuffle_dataset: false\n"
        "train_dataloader_shuffle: false\n"
        "length_grouped_sampling: false\n"
        "```\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
