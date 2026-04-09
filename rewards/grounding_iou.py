"""Visual grounding IoU reward — bbox intersection-over-union vs ground truth.

Pairs with `grounding_format.py` and the `grounding_bbox` RL environment.
Extracts a bbox from `<bbox>x1,y1,x2,y2</bbox>` in the completion and
returns its IoU with the ground-truth bbox in the dataset's `bbox_gt`
column. Returns 0.0 if extraction fails.

Dataset must have a `bbox_gt` column where each row is a list of 4 numbers
[x1, y1, x2, y2] in absolute pixel coordinates.
"""

import re

BBOX_RE = re.compile(r"<bbox>\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*</bbox>")


def _iou(pred: tuple[float, float, float, float], gt: tuple[float, float, float, float]) -> float:
    """Standard axis-aligned bbox IoU. Boxes are (x1, y1, x2, y2)."""
    px1, py1, px2, py2 = pred
    gx1, gy1, gx2, gy2 = gt
    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    pred_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    gt_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union = pred_area + gt_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _extract_bbox(text: str) -> tuple[float, float, float, float] | None:
    match = BBOX_RE.search(text)
    if not match:
        return None
    try:
        return tuple(float(x) for x in match.groups())  # type: ignore[return-value]
    except ValueError:
        return None


def grounding_iou_reward(completions, bbox_gt, **kwargs) -> list[float]:
    """IoU between the predicted bbox in the completion and `bbox_gt`."""
    rewards = []
    for completion, gt in zip(completions, bbox_gt, strict=True):
        if isinstance(completion, list):
            text = completion[0].get("content", "")
        else:
            text = str(completion)
        pred = _extract_bbox(text)
        if pred is None:
            rewards.append(0.0)
            continue
        gt_tuple = tuple(float(x) for x in gt)  # type: ignore[arg-type]
        rewards.append(_iou(pred, gt_tuple))
    return rewards
