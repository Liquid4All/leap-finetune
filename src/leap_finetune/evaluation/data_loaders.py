"""Format-agnostic data loading for benchmarks (JSONL, JSON, Parquet, CSV).

Auto-detects format from file extension. Applies the same normalization as
the training pipeline (column renames, JSON parsing, image_root prepending)
so benchmark data uses one unified schema::

    samples = load_benchmark_samples("/data/eval.parquet", limit=500)
"""

import json
import logging
from pathlib import Path

from leap_finetune.data_loaders.validate_loader import normalize_columns

logger = logging.getLogger(__name__)


def load_benchmark_samples(
    path: str,
    limit: int | None = None,
    format: str | None = None,
    image_root: str | None = None,
) -> list[dict]:
    """Load and normalize benchmark samples from any supported format.

    Applies the same normalization as the training pipeline: renames column
    aliases (``conversation`` -> ``messages``), parses JSON strings, and
    prepends ``image_root`` to relative image paths.
    """
    if format is None:
        format = _detect_format(path)

    loader = _FORMAT_LOADERS.get(format)
    if loader is None:
        raise ValueError(
            f"Unknown format: {format!r}. Available: {sorted(_FORMAT_LOADERS)}"
        )

    samples = loader(path, limit)

    # Reuse training pipeline normalization: column renames, JSON parsing, image_root
    normalize = normalize_columns("vlm_sft", image_root=image_root)
    samples = [normalize(s) for s in samples]

    logger.info("Loaded %d samples from %s (format=%s)", len(samples), path, format)
    return samples


def _detect_format(path: str) -> str:
    p = path.lower()
    if p.endswith((".jsonl", ".ndjson")):
        return "jsonl"
    if p.endswith(".json"):
        return "json"
    if p.endswith((".parquet", ".pq")):
        return "parquet"
    if p.endswith(".csv"):
        return "csv"
    if Path(path).is_dir():
        return "parquet"
    return "jsonl"


def _load_jsonl(path: str, limit: int | None) -> list[dict]:
    samples = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            samples.append(json.loads(line))
    return samples


def _load_json(path: str, limit: int | None) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON benchmark file must contain a top-level array")
    return data[:limit] if limit else data


def _load_parquet(path: str, limit: int | None) -> list[dict]:
    import glob as glob_mod

    import pandas as pd

    p = Path(path)
    if p.is_dir():
        files = sorted(glob_mod.glob(str(p / "*.parquet")))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    else:
        df = pd.read_parquet(path)
    if limit:
        df = df.head(limit)
    return df.to_dict("records")


def _load_csv(path: str, limit: int | None) -> list[dict]:
    import pandas as pd

    df = pd.read_csv(path)
    if limit:
        df = df.head(limit)
    return df.to_dict("records")


_FORMAT_LOADERS: dict[str, callable] = {
    "jsonl": _load_jsonl,
    "json": _load_json,
    "parquet": _load_parquet,
    "csv": _load_csv,
}
