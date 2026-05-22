"""Canonical prompt templates for visual grounding (REC + multi-bbox).

Mined from a ~5K-row sample of a grounding pretraining mix and split
into two pools:

  * ``FORMAT_HINTS``: the JSON bbox output spec, prepended to the
    natural task prompt. We mix them into 100% of cookbook prompts
    because LFM2.5-VL has NOT been pretrained on grounding, so the
    output format needs to be taught explicitly.
  * ``CANONICAL_REC_PROMPTS``: natural single-target referring
    expression task prompts. The model is supposed to output a JSON
    array (single-element for one target, multi-element when multiple
    boxes apply).

Usage::

    import random
    rng = random.Random(seed)
    hint   = rng.choice(FORMAT_HINTS)
    prompt = rng.choice(CANONICAL_REC_PROMPTS).format(ref=ref)
    user_text = f"{hint}\\n\\n{prompt}"
"""

from __future__ import annotations

# Canonical fixed JSON-format hint used by EVERY evaluation row across
# every benchmark and every run. Single source of truth — imported by
# both prepare_evals.py (RefCOCO/+/g) and prepare_data.py (mgrounding
# test split). Training rows still sample from FORMAT_HINTS for
# robustness.
EVAL_FORMAT_HINT: str = (
    'Provide the location as JSON: [{"label": "name", "bbox": [x1, y1, x2, y2]}]. '
    "Bounding box coordinates are normalized between 0 and 1."
)

# JSON output format hints, prepended to the natural task prompt.
# Multiple phrasings so the model learns the format spec, not a single string.
# Index 0 MUST equal EVAL_FORMAT_HINT so the canonical eval phrasing is
# also present in the training mix.
FORMAT_HINTS: list[str] = [
    EVAL_FORMAT_HINT,
    'Return the object location as a JSON array: [{"label": "name", "bbox": [x1, y1, x2, y2]}]. '
    "Coordinates are normalized to [0,1].",
    "When locating objects, return their bounding boxes as a JSON array: "
    '[{"label": "object", "bbox": [x1, y1, x2, y2]}]. Coordinates are normalized to [0,1].',
    "Provide object locations as a JSON array of bounding box annotations: "
    '[{"label": "...", "bbox": [x1, y1, x2, y2]}]. Coordinates are between 0 and 1.',
    "Respond with bounding box locations in a JSON array of "
    '{"label": "...", "bbox": [x1, y1, x2, y2]} entries. '
    "Use [0,1]-normalized coordinates where (0,0) is top-left.",
    'Output a JSON array of objects: [{"label": "...", "bbox": [x1, y1, x2, y2]}, ...]. '
    "Multiple objects → multiple JSON entries. Coordinates are normalized to [0,1].",
    "Return one JSON entry per detected object in the format "
    '[{"label": "...", "bbox": [x1, y1, x2, y2]}]. Use [0,1]-normalized coordinates.',
    "For each object you locate, emit a JSON entry of the form "
    '{"label": "...", "bbox": [x1, y1, x2, y2]}, wrapped in a JSON array. '
    "Coordinates are in the [0,1] range.",
]

# Natural single-target REC prompts, drawn from the top of a grounding
# pretraining mix by frequency. Each has a single ``{ref}`` placeholder
# for the referring expression.
CANONICAL_REC_PROMPTS: list[str] = [
    'Identify the region of "{ref}" in the image.',
    "Identify the bounding box of the region described by the following expression: {ref}.",
    "Please locate the bounding box of the region described in this sentence: {ref}.",
    "Please specify the bounding box coordinates for the region indicated by this sentence: {ref}.",
    "Give the bounding box coordinates of the area mentioned in this sentence: {ref}.",
    "Please determine the bounding box coordinates for the area described here: {ref}.",
    "Please find the bounding box coordinates for the area described by: {ref}.",
    "Please identify the bounding box coordinates for the area described by this sentence: {ref}.",
    'Where is "{ref}" in the image?',
    "Give the bounding box of the region that this sentence refers to: {ref}.",
    "Please provide the bounding box coordinate of the region this sentence describes: {ref}.",
    "Provide the bounding box coordinates for the region referred to by this sentence: {ref}.",
    'Find where "{ref}" is in the image.',
    "Provide the bounding box of the region this expression refers to: {ref}.",
    'Locate "{ref}" in the image.',
    "Please carefully check the image and detect the following objects: {ref}.",
    "Provide the bounding box coordinates that match this description: {ref}.",
]
