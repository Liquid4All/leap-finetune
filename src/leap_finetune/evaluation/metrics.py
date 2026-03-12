"""Scoring functions for benchmark evaluation.

Built-in metrics: ``grounding_iou``, ``short_answer``, ``mcq_gen``.

To add a new metric:
  1. Define a function: ``(prediction: str, ground_truth: str, **kwargs) -> float``
  2. Add it to ``_METRIC_DISPATCH`` below.
  3. Add the metric name to ``GENERATION_METRICS`` or ``LOGPROB_METRICS`` in ``vlm_config.py``.
"""

import ast
import json
import re


# -- Built-in scoring functions --


def score_grounding_iou(
    prediction: str, ground_truth: str, iou_threshold: float = 0.5, **_
) -> float:
    """1.0 if predicted bbox overlaps ground truth above threshold, else 0.0."""
    pred_bbox = _parse_bbox(prediction)
    gt_bbox = _parse_bbox(ground_truth)

    if pred_bbox is None or gt_bbox is None:
        return 0.0

    return 1.0 if _compute_iou(pred_bbox, gt_bbox) >= iou_threshold else 0.0


def score_short_answer(
    prediction: str, ground_truth: str, match_mode: str = "contains", **_
) -> float:
    """1.0 if ground truth appears in the prediction (case-insensitive).

    match_mode="any_in_array": ground truth is a JSON array — match if any element
    appears in the prediction.
    """
    if (
        match_mode == "any_in_array"
        and ground_truth.startswith("[")
        and ground_truth.endswith("]")
    ):
        gt_clean = ground_truth.replace('""', '"')
        pred_lower = prediction.lower().strip().replace("\n", " ")
        try:
            gt_array = ast.literal_eval(gt_clean)
        except (ValueError, SyntaxError):
            gt_array = [ground_truth]

        for gt_item in gt_array:
            if str(gt_item).lower().strip().replace("\n", " ") in pred_lower:
                return 1.0
        return 0.0

    return 1.0 if ground_truth.lower().strip() in prediction.lower().strip() else 0.0


def score_mcq_gen(prediction: str, ground_truth: str, **_) -> float:
    """1.0 if extracted MCQ letter matches ground truth."""
    if prediction.strip() == ground_truth.strip():
        return 1.0

    gt_letter = _extract_mcq_answer(ground_truth)
    pred_letter = _extract_mcq_answer(prediction)

    if gt_letter is None or pred_letter is None:
        return 0.0

    return 1.0 if gt_letter == pred_letter else 0.0


# -- Metric dispatch (add new metrics here) --

_METRIC_DISPATCH: dict[str, callable] = {
    "grounding_iou": score_grounding_iou,
    "short_answer": score_short_answer,
    "mcq_gen": score_mcq_gen,
}


def compute_metric(
    metric_type: str, prediction: str, ground_truth: str, **kwargs
) -> float:
    """Look up and call the named scoring function."""
    fn = _METRIC_DISPATCH.get(metric_type)
    if fn is None:
        raise ValueError(
            f"Unknown metric: {metric_type!r}. Available: {sorted(_METRIC_DISPATCH)}"
        )
    return fn(prediction, ground_truth, **kwargs)


# -- Internal helpers --

_VALID_MCQ_LETTERS = set("abcdef")

_MCQ_PATTERNS = [
    r"\b([a-f])[\.\(\):, ]?",
    r"(?:option |choice |answer: |the answer is )\s*([a-f])\b",
    r"^([a-f])[:).]",
    r"\b([a-f])[\.\s]*(?:is correct|is the answer|appears to be right)\b",
]


def _extract_mcq_answer(text: str) -> str | None:
    """Extract a single MCQ letter (A-F) from free-form text."""
    text = text.strip().lower()

    if len(text) == 1 and text in _VALID_MCQ_LETTERS:
        return text.upper()

    for pattern in _MCQ_PATTERNS:
        matches = re.findall(pattern, text)
        valid = [m for m in matches if m in _VALID_MCQ_LETTERS]
        if len(valid) == 1:
            return valid[0].upper()
    return None


def _parse_bbox(text: str) -> list[float] | None:
    """Parse bounding box [x1, y1, x2, y2] from JSON or Python literal formats.

    Handles nested structures like [{"bbox": [...]}] and normalizes 0-1000 coords to 0-1.
    """
    if not isinstance(text, str):
        return None

    text = text.strip()
    json_match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            data = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None

    bbox = None
    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], dict):
            for item in data:
                if "bbox" in item:
                    bbox = item["bbox"]
                    break
        elif isinstance(data[0], list) and len(data[0]) == 4:
            bbox = data[0]
        elif len(data) == 4 and all(isinstance(x, (int, float)) for x in data):
            bbox = data
    elif isinstance(data, dict) and "bbox" in data:
        bbox = data["bbox"]

    if bbox is None or len(bbox) != 4:
        return None

    try:
        bbox = [float(x) for x in bbox]
    except (ValueError, TypeError):
        return None

    # Normalize 0-1000 coordinate space to 0-1
    if max(abs(c) for c in bbox) > 1.5:
        bbox = [c / 1000.0 for c in bbox]

    return bbox


def _compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute intersection-over-union between two [x1, y1, x2, y2] boxes."""
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b

    # Handle inverted coordinates
    if x1_a > x2_a:
        x1_a, x2_a = x2_a, x1_a
    if y1_a > y2_a:
        y1_a, y2_a = y2_a, y1_a
    if x1_b > x2_b:
        x1_b, x2_b = x2_b, x1_b
    if y1_b > y2_b:
        y1_b, y2_b = y2_b, y1_b

    x1 = max(x1_a, x1_b)
    y1 = max(y1_a, y1_b)
    x2 = min(x2_a, x2_b)
    y2 = min(y2_a, y2_b)

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0
