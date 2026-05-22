"""Build async-eval jsonl files for the grounding cookbook.

Produces three jsonl files for the canonical RefCOCO trio (referring
expression comprehension — REC):

  * ``refcoco_val.jsonl``      — RefCOCO val      (jxu124/refcoco)
  * ``refcoco_plus_val.jsonl`` — RefCOCO+ val     (jxu124/refcocoplus)
  * ``refcocog_val.jsonl``     — RefCOCOg val     (jxu124/refcocog)

Uses the proper REC structure (one referring expression → one bbox),
NOT the lmms-lab variant where the image has a circle drawn on the
target and the question is a generic "caption the circled region"
prompt. The jxu124 family is the canonical REC source.

Each val prompt uses a **fixed** template (same for every row, every
benchmark, every run) so scores are deterministic w.r.t. the prompt
text. SFT keeps random sampling for diversity; eval pins one canonical
phrasing to remove prompt-noise as a confound when comparing runs.

The fourth eval — an in-distribution held-out MGrounding slice — is
written by ``prepare_data.py`` as ``grounding_test/test.parquet`` and
plugged directly into the YAML's ``benchmarks`` block; no separate
script is needed.

Image resolution: jxu124 datasets ship ``image_id`` + ``file_name``
only (no image bytes), so we resolve to local COCO 2014 train images
at ``--coco-train2014``. Download once from
``http://images.cocodataset.org/zips/train2014.zip``.

Usage::

    uv run python cookbook/visual-grounding/prepare_evals.py \\
        --output ./job_datasets/grounding-cookbook/data/grounding_evals \\
        [--limit 500] [--coco-train2014 <path>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Shared canonical hint lives next to this script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt_templates import EVAL_FORMAT_HINT  # noqa: E402

# Override with --coco-train2014. No useful default since COCO 2014 is
# a one-off download, location varies per cluster.
DEFAULT_COCO_TRAIN2014 = Path("./coco/train2014")

# Fixed canonical eval prompt — one of the SFT format-hint variants
# paired with one of the SFT REC task variants. Identical across all
# val benchmarks and every regen of the eval jsonl, so the only thing
# changing between runs is the model checkpoint.
# EVAL_FORMAT_HINT is imported from prompt_templates so prepare_data.py
# (test split) and prepare_evals.py (RefCOCO/+/g) emit the IDENTICAL hint.
# Picked as the winner from a 17-template sweep on refcoco_val (acc 0.648
# vs 0.508 for the worst). The "bounding box coordinates" phrasing matches
# MGrounding's training distribution.
EVAL_TASK_TEMPLATE = (
    "Please find the bounding box coordinates for the area described by: {ref}."
)


def _build_user_text(ref: str) -> str:
    return f"{EVAL_FORMAT_HINT}\n\n{EVAL_TASK_TEMPLATE.format(ref=ref)}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Max samples per benchmark (default: 500).",
    )
    p.add_argument(
        "--coco-train2014",
        type=Path,
        default=DEFAULT_COCO_TRAIN2014,
        help="Path to COCO 2014 train images directory.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _bbox_xywh_pixels_to_xyxy_norm(
    bbox: list[float], w: int, h: int
) -> list[float] | None:
    x, y, bw, bh = bbox
    xyxy = [
        round(x / w, 4),
        round(y / h, 4),
        round((x + bw) / w, 4),
        round((y + bh) / h, 4),
    ]
    if not (xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1]):
        return None
    return xyxy


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _build_refcoco_variant(
    hf_id: str,
    *,
    split: str,
    coco_root: Path,
    out_path: Path,
    limit: int,
    seed: int,
) -> None:
    from datasets import load_dataset

    del seed  # eval prompt is fixed; row order is the dataset's own order
    print(f"[evals] loading {hf_id} split={split}")
    ds = load_dataset(hf_id, split=split)
    rows: list[dict] = []
    skipped_missing = 0
    skipped_other = 0
    for sample in ds:
        if len(rows) >= limit:
            break
        sentences = sample.get("sentences") or []
        if not sentences:
            skipped_other += 1
            continue

        raw_anns = sample.get("raw_anns")
        raw_image_info = sample.get("raw_image_info")
        if isinstance(raw_anns, str):
            raw_anns = json.loads(raw_anns)
        if isinstance(raw_image_info, str):
            raw_image_info = json.loads(raw_image_info)

        bbox_xywh = (raw_anns or {}).get("bbox") or []
        w = (raw_image_info or {}).get("width")
        h = (raw_image_info or {}).get("height")
        file_name = (raw_image_info or {}).get("file_name")
        if not file_name or not w or not h or len(bbox_xywh) != 4:
            skipped_other += 1
            continue

        xyxy = _bbox_xywh_pixels_to_xyxy_norm(bbox_xywh, w, h)
        if xyxy is None:
            skipped_other += 1
            continue

        img_path = coco_root / file_name
        if not img_path.is_file():
            skipped_missing += 1
            continue

        # First referring expression is typically the cleanest in RefCOCO.
        ref_obj = sentences[0]
        if isinstance(ref_obj, dict):
            ref = (ref_obj.get("raw") or ref_obj.get("sent") or "").strip().rstrip(".")
        else:
            ref = str(ref_obj).strip().rstrip(".")
        if not ref:
            skipped_other += 1
            continue

        user_text = _build_user_text(ref)

        answer = [{"label": ref, "bbox": xyxy}]
        rows.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(img_path)},
                            {"type": "text", "text": user_text},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(answer, ensure_ascii=False),
                            },
                        ],
                    },
                ]
            }
        )

    _write_jsonl(rows, out_path)
    print(
        f"[evals] wrote {len(rows):,} {hf_id} → {out_path} "
        f"(skipped {skipped_missing:,} missing-image, {skipped_other:,} other)"
    )


def main() -> int:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if not args.coco_train2014.is_dir():
        raise SystemExit(
            f"COCO 2014 train images not found at {args.coco_train2014}. "
            f"Pass --coco-train2014 <your-path> or download from "
            f"http://images.cocodataset.org/zips/train2014.zip"
        )

    variants = [
        ("jxu124/refcoco", "validation", "refcoco_val.jsonl"),
        ("jxu124/refcocoplus", "validation", "refcoco_plus_val.jsonl"),
        ("jxu124/refcocog", "validation", "refcocog_val.jsonl"),
    ]
    for hf_id, split, jsonl_name in variants:
        _build_refcoco_variant(
            hf_id,
            split=split,
            coco_root=args.coco_train2014,
            out_path=args.output / jsonl_name,
            limit=args.limit,
            seed=args.seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
