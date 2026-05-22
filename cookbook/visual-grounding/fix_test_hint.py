"""One-off: rewrite an existing grounding test parquet so every row's
leading format-hint is the canonical ``EVAL_FORMAT_HINT``.

Run when the test parquet was produced by an older ``prepare_data.py``
that sampled a random hint per row. After this script, all rows use
the SAME hint as ``prepare_evals.py`` (RefCOCO/+/g) — eval scores
become directly comparable across rows, regens, and runs.

Usage::

    python cookbook/visual-grounding/fix_test_hint.py \\
        --parquet ./job_datasets/grounding-cookbook/data/grounding_test/test.parquet

Writes back to the same path (atomic via temp file).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import _canonicalize_test_hint  # noqa: E402
from prompt_templates import EVAL_FORMAT_HINT  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--parquet",
        required=True,
        type=Path,
        help="Path to the existing test parquet to rewrite in-place.",
    )
    args = p.parse_args()

    src = args.parquet
    if not src.is_file():
        raise SystemExit(f"not found: {src}")

    table = pq.read_table(src)
    rows = [
        dict(zip(table.column_names, vals))
        for vals in zip(*[c.to_pylist() for c in table.columns])
    ]
    print(f"[fix] loaded {len(rows):,} rows from {src}")

    rewritten = 0
    skipped = 0
    for r in rows:
        before = r["messages"]
        _canonicalize_test_hint(r)
        if r["messages"] != before:
            rewritten += 1
        else:
            skipped += 1

    out_table = pa.Table.from_pylist(rows)
    tmp = src.with_suffix(".parquet.tmp")
    pq.write_table(out_table, tmp)
    tmp.replace(src)

    print(
        f"[fix] rewrote hint on {rewritten:,} rows, skipped {skipped:,} (already canonical or malformed)\n"
        f"[fix] canonical hint:\n  {EVAL_FORMAT_HINT}\n"
        f"[fix] wrote {src}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
