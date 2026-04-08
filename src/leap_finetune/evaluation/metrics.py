import ast
import json
import math
import re
from collections import Counter


# === Built-in scoring functions ===


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


def score_bleu(prediction: str, ground_truth: str, max_n: int = 4, **_) -> float:
    """Sentence-level BLEU score (0-1) up to max_n-grams with brevity penalty."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(ground_truth)

    if not pred_tokens or not ref_tokens:
        return 0.0

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / len(pred_tokens))) if pred_tokens else 0.0

    # n-gram precisions with +1 smoothing (smoothing method 1)
    log_avg = 0.0
    for n in range(1, max_n + 1):
        pred_ngrams = _get_ngrams(pred_tokens, n)
        ref_ngrams = _get_ngrams(ref_tokens, n)
        clipped = sum(min(pred_ngrams[ng], ref_ngrams[ng]) for ng in pred_ngrams)
        total = max(sum(pred_ngrams.values()), 1)
        # Add-1 smoothing for n > 1 to avoid zero precision killing the score
        if n > 1:
            clipped += 1
            total += 1
        precision = clipped / total
        if precision == 0:
            return 0.0
        log_avg += math.log(precision) / max_n

    return bp * math.exp(log_avg)


def score_rouge_l(prediction: str, ground_truth: str, **_) -> float:
    """ROUGE-L F1 score (0-1) based on longest common subsequence."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(ground_truth)

    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs_len = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs_len / len(pred_tokens)
    recall = lcs_len / len(ref_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# === Metric dispatch (add new metrics here) ===

_METRIC_DISPATCH: dict[str, callable] = {
    "grounding_iou": score_grounding_iou,
    "short_answer": score_short_answer,
    "mcq_gen": score_mcq_gen,
    "bleu": score_bleu,
    "rouge_l": score_rouge_l,
}


def compute_metric(
    metric_type: str, prediction: str, ground_truth: str, **kwargs
) -> float:
    """Dispatch to the named scoring function and return a 0-1 score."""
    fn = _METRIC_DISPATCH.get(metric_type)
    if fn is None:
        raise ValueError(
            f"Unknown metric: {metric_type!r}. Available: {sorted(_METRIC_DISPATCH)}"
        )
    return fn(prediction, ground_truth, **kwargs)


# === Internal helpers ===

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


def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenization for BLEU/ROUGE."""
    return text.lower().split()


def _get_ngrams(tokens: list[str], n: int) -> Counter:
    """Count n-grams in a token list."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence (space-optimised)."""
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for ai in a:
        curr = [0] * (len(b) + 1)
        for j, bj in enumerate(b):
            curr[j + 1] = prev[j] + 1 if ai == bj else max(prev[j + 1], curr[j])
        prev = curr
    return prev[-1]


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
