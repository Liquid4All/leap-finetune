"""Convert MGrounding-630k (HuggingFace) into leap-finetune parquet,
then deterministically split into SFT and GRPO held-out pools.

Source: ``Michael4933/MGrounding-630k`` (Migician, ACL 2025).

Conversion behavior:
  * Drops Object_Tracking samples (per-frame sequential bboxes don't
    match single-shot JSON-list output).
  * Iterates each conversation's (human, gpt) pairs — samples are
    typically multi-turn, so one input row can produce 0..N output rows.
  * Extracts (label, bbox) pairs from gpt turns:
      - bbox normalized from MGrounding's 0-1000 coords to 0-1.
      - label = nearest preceding ``<|object_ref_start|>...<|object_ref_end|>``
        in the gpt turn; else extracted from the user turn; else "target".
  * Preserves multi-image input (all images in the user turn).
  * Assistant output is the LFM2-VL native bbox JSON:
    ``[{"label": "...", "bbox": [x1, y1, x2, y2]}, ...]``.

The output schema (single ``messages`` column, JSON-encoded conversation)
drives **both** ``vlm_sft`` and ``vlm_grpo`` training — leap-finetune's
``vlm_grpo`` loader auto-splits ``messages`` into ``prompt`` (user turns)
and ``solution`` (last assistant turn = ground-truth bbox JSON). So the
same parquet underlies both phases; GRPO simply trains on a held-out
slice the SFT run never saw.

Layout produced under ``--output``::

    grounding_sft/train.parquet     # 65% (default) — drives SFT
    grounding_grpo/train.parquet    # 25% (default) — held out from SFT
    grounding_test/test.parquet     # 10% (default) — held out from both;
                                    # async-eval reads this as a benchmark.

Usage::

    python cookbook/visual-grounding/prepare_data.py \\
        --output ./job_datasets/grounding-cookbook/data \\
        [--grpo-fraction 0.25] [--test-fraction 0.10] [--limit 500000] \\
        [--cache-dir ./job_datasets/grounding-cookbook/cache]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from prompt_templates import EVAL_FORMAT_HINT, FORMAT_HINTS

REPO_ID = "Michael4933/MGrounding-630k"

# Module-level RNG seeded in main() — used to draw a format-hint per row
# deterministically so a re-run gives identical output.
_HINT_RNG = random.Random()

# Object_Tracking emits per-frame sequential bboxes — different output shape
# than our single-shot JSON-list, so it'd pollute SFT signal.
SKIP_SUBSET = "Object_Tracking"

_IMAGE_TAG = re.compile(r"<image>\s*")
_REF_TAG = re.compile(r"<\|object_ref_start\|>(.+?)<\|object_ref_end\|>", re.DOTALL)
_BOX_TAG = re.compile(
    r"<\|box_start\|>\(([\d.]+),([\d.]+)\),\(([\d.]+),([\d.]+)\)<\|box_end\|>"
)
# Common patterns for extracting a referring expression from user text
# when no <|object_ref|> tag is present. Ordered by specificity.
_FALLBACK_PATTERNS = [
    re.compile(r"refers to:?\s*\"?([^\"\n.]+?)[\".]?$", re.IGNORECASE),
    re.compile(r"region of\s+\"?([^\"\n.]+?)[\".]?\s+in", re.IGNORECASE),
    re.compile(
        r"(?:locate|find|identify|detect)\s+\"?([^\"\n.?:]+?)[\".?]?$", re.IGNORECASE
    ),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output dir. Contains grounding_sft/ and grounding_grpo/.",
    )
    p.add_argument(
        "--grpo-fraction",
        type=float,
        default=0.25,
        help="Fraction held out for GRPO (default: 0.25).",
    )
    p.add_argument(
        "--test-fraction",
        type=float,
        default=0.10,
        help="Fraction held out as async-eval test set (default: 0.10).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap total output samples after conversion. Default: full.",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="HF cache dir. Default: $HF_HOME or ~/.cache/huggingface.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed — split is fully deterministic.",
    )
    p.add_argument(
        "--skip-image-check",
        action="store_true",
        help="Don't require image files to exist on disk. Useful for smoke-testing the parser before snapshot_download completes.",
    )
    return p.parse_args()


def _subset_of(images: list) -> str:
    """Subset name is the first directory under ``./MGrounding-630k/`` in the
    first image path. Returns '<unknown>' if it can't be derived."""
    if not images:
        return "<unknown>"
    first = images[0] if isinstance(images[0], str) else images[0].get("image", "")
    parts = first.split("/")
    if len(parts) >= 3 and parts[1] == "MGrounding-630k":
        return parts[2]
    return "<unknown>"


def _strip_image_tags(text: str) -> str:
    return _IMAGE_TAG.sub("", text).strip()


def _user_fallback_label(user_text: str) -> str | None:
    """Extract a referring expression from the user turn when the gpt turn
    has no <|object_ref|> tag. Tries explicit ref tags first, then a few
    common natural-language patterns."""
    refs = _REF_TAG.findall(user_text)
    if refs:
        return refs[0].strip().rstrip(".")
    for pat in _FALLBACK_PATTERNS:
        m = pat.search(user_text)
        if m:
            candidate = m.group(1).strip().rstrip(".,:;").strip()
            if 2 <= len(candidate) <= 120:
                return candidate
    return None


def _extract_pairs(gpt_text: str, fallback_label: str) -> list[dict]:
    """Find (label, bbox) pairs in a gpt response.

    Label = nearest preceding ``<|object_ref|>`` tag in the gpt turn;
    else ``fallback_label`` (from user turn) if not None; else ``target``.
    """
    pairs: list[dict] = []
    refs = [
        (m.end(), m.group(1).strip().rstrip(".")) for m in _REF_TAG.finditer(gpt_text)
    ]
    for b in _BOX_TAG.finditer(gpt_text):
        x1, y1, x2, y2 = (float(v) / 1000.0 for v in b.groups())
        if not (x2 > x1 and y2 > y1 and all(0.0 <= v <= 1.0 for v in (x1, y1, x2, y2))):
            continue
        label = fallback_label or "target"
        best = -1
        for end, txt in refs:
            if end <= b.start() and end > best:
                best = end
                label = txt
        pairs.append(
            {
                "label": label,
                "bbox": [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)],
            }
        )
    return pairs


def _resolve_image_path(image_ref, dataset_root: Path) -> Path:
    rel = image_ref if isinstance(image_ref, str) else image_ref.get("image", "")
    return (dataset_root / rel).resolve()


def _canonicalize_test_hint(row: dict) -> dict:
    """Rewrite a row's user text so the leading format-hint is EVAL_FORMAT_HINT.

    The user text was built as ``f"{hint}\\n\\n{task}"``. We split on the
    first ``\\n\\n`` and replace the prefix. If the row is malformed
    (no ``\\n\\n`` separator, multiple text segments, etc.), it's left
    as-is and we count it. Idempotent: re-running on an already-canonical
    row is a no-op.
    """
    msgs = json.loads(row["messages"])
    user = msgs[0]
    text_segs = [s for s in user.get("content", []) if s.get("type") == "text"]
    if len(text_segs) != 1:
        return row
    body = text_segs[0]["text"]
    if "\n\n" not in body:
        return row
    _, task = body.split("\n\n", 1)
    text_segs[0]["text"] = f"{EVAL_FORMAT_HINT}\n\n{task}"
    row["messages"] = json.dumps(msgs, ensure_ascii=False)
    return row


def _convert_one(
    sample: dict,
    dataset_root: Path,
    *,
    skip_image_check: bool,
) -> list[dict]:
    """Returns 0..N converted samples (one per qualifying (human, gpt) pair)."""
    images = sample.get("images") or []
    conversations = sample.get("conversations") or []
    if not images or len(conversations) < 2:
        return []

    subset = _subset_of(images)
    if subset == SKIP_SUBSET:
        return []

    image_paths = [_resolve_image_path(i, dataset_root) for i in images]
    if not skip_image_check:
        if not all(p.is_file() for p in image_paths):
            return []
    image_paths_str = [str(p) for p in image_paths]

    out: list[dict] = []
    # Walk (human, gpt) pairs.
    for i in range(0, len(conversations) - 1, 2):
        h, g = conversations[i], conversations[i + 1]
        if h.get("from") != "human" or g.get("from") != "gpt":
            continue

        user_text = _strip_image_tags(h.get("value", ""))
        # Strip ref tags but keep inner content, so user text stays readable.
        user_text_clean = _REF_TAG.sub(r"\1", user_text)
        fallback = _user_fallback_label(user_text)
        pairs = _extract_pairs(g.get("value", ""), fallback)
        if not pairs:
            continue

        # Prepend a sampled format-hint preamble so the model learns the
        # JSON bbox output spec explicitly. LFM2.5-VL was not pretrained
        # on grounding, so implicit format learning isn't enough.
        hint = _HINT_RNG.choice(FORMAT_HINTS)
        user_text_with_hint = f"{hint}\n\n{user_text_clean}"

        user_content = [{"type": "image", "image": p} for p in image_paths_str] + [
            {"type": "text", "text": user_text_with_hint}
        ]

        messages = [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": json.dumps(pairs, ensure_ascii=False)},
                ],
            },
        ]
        out.append({"messages": json.dumps(messages, ensure_ascii=False)})
    return out


def _download_dataset(cache_dir: Path | None) -> tuple[Path, Path]:
    """Returns (snapshot_root, image_root_parent).

    ``image_root_parent`` is the directory that ``./MGrounding-630k/...``
    relative paths in the manifest resolve against — i.e. the parent of
    the actual on-disk MGrounding-630k root produced by extraction.
    """
    from huggingface_hub import snapshot_download

    print(f"[prep] snapshot_download {REPO_ID} (~140 GB, one-time)")
    local = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    snapshot_root = Path(local)
    real_mg_root = _ensure_zips_extracted(snapshot_root)
    print(f"[prep] image root: {real_mg_root}")
    return snapshot_root, real_mg_root.parent


def _ensure_zips_extracted(snapshot_root: Path) -> Path:
    """MGrounding ships per-subset ``<Subset>.zip``; extract each in place
    if its target directory isn't already populated. Object_Tracking is
    skipped entirely — we filter it out at conversion time anyway, no
    point burning ~30 GB extracting it.

    Group_Grounding (and Object_Tracking, were we to extract it) is a
    multi-volume ZIP (.z01, .z02, .z03, .zip) which Python's stdlib
    ``zipfile`` cannot read. Use ``7z`` for those.

    Quirk: the zips were authored against the original author's machine
    layout, so files land under
    ``<snapshot>/home/liyou/opensource_datasets/MGrounding-630k/...``.
    Returns the path of that actual on-disk MGrounding-630k root, which
    callers use to resolve manifest image paths.
    """
    import shutil
    import subprocess
    import zipfile

    seven_zip = shutil.which("7z") or shutil.which("7zz") or shutil.which("7za")

    zips = sorted(snapshot_root.glob("*.zip"))

    for z in zips:
        subset = z.stem
        if subset == SKIP_SUBSET:
            print(f"[prep] skipping {z.name} (filtered at conversion)")
            continue
        if _find_subset_dir(snapshot_root, subset) is not None:
            print(f"[prep] {subset} already extracted, skipping")
            continue
        is_multipart = any(snapshot_root.glob(f"{subset}.z[0-9][0-9]"))
        print(
            f"[prep] extracting {z.name} ({'multipart→' + Path(seven_zip).name if is_multipart else 'zipfile'}) ..."
        )
        if is_multipart:
            if seven_zip is None:
                raise RuntimeError(
                    f"Need a 7-Zip CLI ('7z', '7zz', or '7za') in PATH to "
                    f"extract the multi-volume archive {z.name}. Install or "
                    f"drop a binary on $PATH and re-run."
                )
            subprocess.run(
                [seven_zip, "x", "-y", f"-o{snapshot_root}", str(z)],
                check=True,
            )
        else:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(snapshot_root)

    # Locate the actual MGrounding-630k root (somewhere under snapshot_root,
    # not necessarily at the top level).
    candidates = [d for d in snapshot_root.rglob("MGrounding-630k") if d.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            f"No 'MGrounding-630k' directory found under {snapshot_root} after extraction."
        )
    # Pick the one whose children look like subsets (Common_Object, etc.).
    real = max(candidates, key=lambda d: sum(1 for _ in d.iterdir()))
    return real


def _find_subset_dir(snapshot_root: Path, subset: str) -> Path | None:
    """Return path of an extracted-and-populated <subset> directory, or None."""
    for c in snapshot_root.rglob(subset):
        if c.is_dir() and any(c.iterdir()):
            # Skip dirs that aren't direct children of an MGrounding-630k root.
            if c.parent.name == "MGrounding-630k":
                return c
    return None


def _load_manifest(snapshot_root: Path):
    candidates = sorted(snapshot_root.rglob("MGrounding-630k.json"))
    if not candidates:
        candidates = sorted(snapshot_root.rglob("MGrounding*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No MGrounding-630k JSON under {snapshot_root}. snapshot_download didn't complete?"
        )
    print(f"[prep] loading manifest {candidates[0]}")
    with candidates[0].open() as f:
        return json.load(f)


def _write_parquet(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=pa.schema([("messages", pa.string())])),
        path,
    )


def main() -> int:
    args = _parse_args()
    if not (0.0 < args.grpo_fraction < 1.0):
        raise SystemExit("--grpo-fraction must be in (0, 1)")
    if not (0.0 < args.test_fraction < 1.0):
        raise SystemExit("--test-fraction must be in (0, 1)")
    if args.grpo_fraction + args.test_fraction >= 1.0:
        raise SystemExit("--grpo-fraction + --test-fraction must be < 1.0")
    random.seed(args.seed)
    # Per-sample format-hint sampling is deterministic for the same seed.
    _HINT_RNG.seed(args.seed)

    if args.skip_image_check:
        # Smoke-test path: only need the manifest, not the 140 GB of zips.
        from huggingface_hub import hf_hub_download

        manifest_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="MGrounding-630k.json",
            repo_type="dataset",
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
        )
        # Image paths won't resolve in smoke mode anyway; use snapshot root.
        image_root_parent = Path(manifest_path).parent
        with open(manifest_path) as f:
            samples = json.load(f)
    else:
        snapshot_root, image_root_parent = _download_dataset(args.cache_dir)
        samples = _load_manifest(snapshot_root)

    print(f"[prep] {len(samples):,} raw rows in manifest")

    converted: list[dict] = []
    skipped_raw = 0
    for s in samples:
        outs = _convert_one(
            s, image_root_parent, skip_image_check=args.skip_image_check
        )
        if not outs:
            skipped_raw += 1
            continue
        converted.extend(outs)
        if args.limit and len(converted) >= args.limit:
            converted = converted[: args.limit]
            break

    print(f"[prep] {skipped_raw:,} input rows dropped; {len(converted):,} output rows")

    random.shuffle(converted)
    n_total = len(converted)
    n_test = int(round(n_total * args.test_fraction))
    n_grpo = int(round(n_total * args.grpo_fraction))
    test_pool = converted[:n_test]
    grpo_pool = converted[n_test : n_test + n_grpo]
    sft_pool = converted[n_test + n_grpo :]

    # Test rows: replace the per-row sampled hint with the canonical
    # EVAL_FORMAT_HINT so mgrounding_test sees the SAME hint as
    # RefCOCO/+/g across rows and across regens. Train + GRPO keep the
    # sampled-hint diversity (helps SFT robustness).
    test_pool = [_canonicalize_test_hint(row) for row in test_pool]

    sft_path = args.output / "grounding_sft" / "train.parquet"
    grpo_path = args.output / "grounding_grpo" / "train.parquet"
    test_path = args.output / "grounding_test" / "test.parquet"
    _write_parquet(sft_pool, sft_path)
    _write_parquet(grpo_pool, grpo_path)
    _write_parquet(test_pool, test_path)

    print(
        f"[prep] SFT  pool {len(sft_pool):>7,} → {sft_path}\n"
        f"[prep] GRPO pool {len(grpo_pool):>7,} → {grpo_path}\n"
        f"[prep] TEST pool {len(test_pool):>7,} → {test_path}\n"
        f"[prep] All three pools are pairwise disjoint (seed {args.seed})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
