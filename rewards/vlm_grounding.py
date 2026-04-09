"""VLM visual grounding recipe for RLVR GRPO training.

A :class:`VLMGroundingRecipe` plus its four reward functions, all in one
file so the whole task is easy to read and easy to fork.

Use it from YAML:

.. code-block:: yaml

    rewards:
      recipe: "./rewards/vlm_grounding.py::VLMGroundingRecipe"

To extend (e.g. add an LLM-judge reward for object descriptions), write a
new file in ``rewards/`` that subclasses this recipe — see
``rewards/README.md`` for the pattern.

## Task

Train a VLM to output bounding boxes as structured JSON:

.. code-block:: json

    {"bboxes": [[x1, y1, x2, y2], [x1, y1, x2, y2]]}

The recipe's rewards jointly shape the model to produce valid JSON with
the right schema (``json_format_reward``, ``bbox_schema_reward``),
localize the correct regions (``ciou_reward`` — Complete IoU vs
``bbox_gt``), and handle multi-object scenes with optimal 1-to-1 matching
(``hungarian_reward`` — mean CIoU over a pure-Python Hungarian
assignment).

## Expected dataset columns

* ``prompt`` — the VLM GRPO messages list with the image + instruction.
* ``bbox_gt`` — single ground-truth box ``[x1, y1, x2, y2]`` in absolute
  pixel coordinates. Used by ``ciou_reward``.
* ``bboxes_gt`` — list of ground-truth boxes
  ``[[x1, y1, x2, y2], ...]``. Used by ``hungarian_reward``. If your
  dataset has a single box per sample, set ``bboxes_gt = [bbox_gt]``.

Rewards for missing or malformed predictions return ``0.0`` — they never
raise, so a single bad completion can't crash a training step.
"""

from __future__ import annotations

import json
import re
from typing import Sequence

from leap_finetune.rewards import Recipe

# ---------------------------------------------------------------------------
# Geometry primitives (pure Python — no numpy/scipy dependency)
# ---------------------------------------------------------------------------


def _iou(pred: Sequence[float], gt: Sequence[float]) -> float:
    """Standard axis-aligned bbox IoU. Boxes are ``(x1, y1, x2, y2)``."""
    px1, py1, px2, py2 = pred
    gx1, gy1, gx2, gy2 = gt
    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    pa = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    ga = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union = pa + ga - inter
    if union <= 0:
        return 0.0
    return inter / union


def _ciou(pred: Sequence[float], gt: Sequence[float]) -> float:
    """Complete IoU (Zheng et al., 2020) — IoU plus center-distance and
    aspect-ratio consistency penalties.

    CIoU = IoU - ρ²(b_pred, b_gt) / c² - α · v

    where ρ is the Euclidean distance between centers, c is the diagonal of
    the smallest enclosing axis-aligned box, v is an aspect-ratio consistency
    term, and α is a trade-off weight that activates when IoU is high.

    Returns a value in ``[-1, 1]``. We clip to ``[0, 1]`` so it's usable as
    a direct RL reward (negative rewards for GRPO advantage normalization
    work, but clipping avoids surprising sign flips).
    """
    import math

    iou = _iou(pred, gt)

    px1, py1, px2, py2 = pred
    gx1, gy1, gx2, gy2 = gt

    # Center points
    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
    center_dist_sq = (pcx - gcx) ** 2 + (pcy - gcy) ** 2

    # Smallest enclosing box
    ex1, ey1 = min(px1, gx1), min(py1, gy1)
    ex2, ey2 = max(px2, gx2), max(py2, gy2)
    diag_sq = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2
    if diag_sq <= 0:
        return max(0.0, iou)

    # Aspect ratio term
    pw = max(1e-6, px2 - px1)
    ph = max(1e-6, py2 - py1)
    gw = max(1e-6, gx2 - gx1)
    gh = max(1e-6, gy2 - gy1)
    v = (4.0 / (math.pi**2)) * (math.atan(gw / gh) - math.atan(pw / ph)) ** 2
    alpha = v / (1.0 - iou + v + 1e-6) if iou >= 0.5 else 0.0

    ciou = iou - (center_dist_sq / diag_sq) - alpha * v
    return max(0.0, ciou)


# ---------------------------------------------------------------------------
# Hungarian matching (pure Python — no scipy dependency)
# ---------------------------------------------------------------------------
# We implement a simple exact O(n^4) Hungarian / Kuhn–Munkres for small
# grounding problems (typical: <= 10 boxes). For larger N the customer can
# swap in scipy.optimize.linear_sum_assignment.


def _hungarian_assignment(cost: list[list[float]]) -> list[tuple[int, int]]:
    """Exact Hungarian assignment on an n×m cost matrix (n rows, m cols).

    Returns a list of ``(row, col)`` pairs that minimize total cost, with
    ``min(n, m)`` pairs. Padded rows/cols are ignored in the output.
    """
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0])
    size = max(n, m)
    # Pad to a square matrix with a large value (so pads don't get matched first)
    BIG = 1e9
    a = [[(cost[i][j] if i < n and j < m else BIG) for j in range(size)] for i in range(size)]

    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    INF = float("inf")

    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, size + 1):
                if not used[j]:
                    cur = a[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0 != 0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    pairs: list[tuple[int, int]] = []
    for j in range(1, size + 1):
        if p[j] != 0:
            row, col = p[j] - 1, j - 1
            if row < n and col < m:
                pairs.append((row, col))
    return pairs


# ---------------------------------------------------------------------------
# Completion → bbox extraction
# ---------------------------------------------------------------------------


def _get_text(completion) -> str:
    """Handle both conversational and string completions defensively."""
    if isinstance(completion, list):
        if completion and isinstance(completion[0], dict):
            return completion[0].get("content", "") or ""
        return ""
    return str(completion) if completion is not None else ""


# Match the first balanced JSON object in a block of text. Grounding models
# often wrap their answer in prose, so we can't assume the whole completion
# is JSON — we look for a `{ ... }` substring and try to parse it.
_JSON_OBJ_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    """Extract the first parseable JSON object from ``text``. Returns None
    if no valid JSON object is found.
    """
    # Try the whole string first
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    # Scan for embedded `{ ... }` substrings
    for match in _JSON_OBJ_RE.finditer(text):
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _coerce_bbox(value) -> tuple[float, float, float, float] | None:
    """Try to coerce ``value`` to an ``(x1, y1, x2, y2)`` tuple of floats."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return tuple(float(v) for v in value)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _extract_bboxes(text: str) -> list[tuple[float, float, float, float]]:
    """Parse ``text`` as JSON and return the list of boxes.

    Accepts either ``{"bboxes": [[x1,y1,x2,y2], ...]}`` (multi-object) or
    ``{"bbox": [x1,y1,x2,y2]}`` (single-object). Returns an empty list if
    parsing fails or no valid boxes are present.
    """
    obj = _extract_json(text)
    if obj is None:
        return []
    boxes: list[tuple[float, float, float, float]] = []
    if "bboxes" in obj and isinstance(obj["bboxes"], list):
        for raw in obj["bboxes"]:
            box = _coerce_bbox(raw)
            if box is not None:
                boxes.append(box)
        return boxes
    if "bbox" in obj:
        box = _coerce_bbox(obj["bbox"])
        if box is not None:
            return [box]
    return []


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


def json_format_reward(completions, **kwargs) -> list[float]:
    """1.0 if the completion contains a parseable JSON object, 0.0 otherwise.

    The cheapest early-training signal — teaches the model to stop
    hallucinating prose before we start scoring accuracy. Use a low weight
    (~0.1) so it doesn't dominate the main signal once the model's output is
    already well-formed.
    """
    rewards = []
    for completion in completions:
        text = _get_text(completion)
        rewards.append(1.0 if _extract_json(text) is not None else 0.0)
    return rewards


def bbox_schema_reward(completions, **kwargs) -> list[float]:
    """1.0 if the JSON output has a valid ``bbox`` or ``bboxes`` field with
    the expected shape, 0.0 otherwise. Stronger shape check than
    ``json_format_reward`` — rejects JSON that parses but isn't a bbox answer.
    """
    rewards = []
    for completion in completions:
        text = _get_text(completion)
        boxes = _extract_bboxes(text)
        rewards.append(1.0 if boxes else 0.0)
    return rewards


def ciou_reward(completions, bbox_gt=None, **kwargs) -> list[float]:
    """Complete IoU between the first predicted box and ground truth.

    Dataset must have a ``bbox_gt`` column where each row is
    ``[x1, y1, x2, y2]`` in absolute pixel coordinates. Returns 0.0 when the
    completion has no parseable box or when ``bbox_gt`` is missing for a row.

    Best for single-object grounding. For multi-object scenes use
    ``hungarian_reward`` instead (or stack both).
    """
    if bbox_gt is None:
        return [0.0] * len(completions)
    rewards = []
    for completion, gt in zip(completions, bbox_gt, strict=False):
        text = _get_text(completion)
        boxes = _extract_bboxes(text)
        if not boxes or gt is None:
            rewards.append(0.0)
            continue
        gt_box = _coerce_bbox(gt)
        if gt_box is None:
            rewards.append(0.0)
            continue
        rewards.append(_ciou(boxes[0], gt_box))
    return rewards


def hungarian_reward(completions, bboxes_gt=None, **kwargs) -> list[float]:
    """Mean CIoU over the optimal 1-to-1 Hungarian matching between predicted
    and ground-truth boxes.

    For multi-object grounding. Dataset must have a ``bboxes_gt`` column
    where each row is a list of boxes ``[[x1,y1,x2,y2], ...]``. If your
    dataset has only a single box per sample, wrap it:
    ``bboxes_gt = [[[x1,y1,x2,y2]]]``.

    The reward is penalized for both false positives (predicted boxes with
    no match) and false negatives (missed ground-truth boxes): unmatched
    boxes contribute 0 to the mean, and we divide by
    ``max(len(pred), len(gt))`` so extra or missing boxes always hurt.
    """
    if bboxes_gt is None:
        return [0.0] * len(completions)

    rewards = []
    for completion, gt_list in zip(completions, bboxes_gt, strict=False):
        text = _get_text(completion)
        preds = _extract_bboxes(text)
        if not preds or not gt_list:
            rewards.append(0.0)
            continue

        gts: list[tuple[float, float, float, float]] = []
        for raw in gt_list:
            box = _coerce_bbox(raw)
            if box is not None:
                gts.append(box)
        if not gts:
            rewards.append(0.0)
            continue

        # Cost = 1 - CIoU (so Hungarian minimizes the cost = maximizes CIoU)
        cost = [[1.0 - _ciou(p, g) for g in gts] for p in preds]
        pairs = _hungarian_assignment(cost)
        total = sum(1.0 - cost[i][j] for i, j in pairs)
        # Penalize extra/missing boxes by dividing by max(pred, gt)
        denom = max(len(preds), len(gts))
        rewards.append(total / denom if denom > 0 else 0.0)
    return rewards


# ---------------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------------


class VLMGroundingRecipe(Recipe):
    """RLVR recipe for VLM visual grounding with bounding boxes.

    Override :meth:`rewards` to change the reward set or weights. The
    weakest shape checks (format + schema) carry tiny weights so they
    fade to near-irrelevance once the model's output is well-formed, and
    the real signal (CIoU + Hungarian matching) dominates.
    """

    description = (
        "VLM visual grounding — JSON-output bounding boxes scored by CIoU "
        "and Hungarian matching against ground truth."
    )

    required_columns = ("prompt", "bbox_gt", "bboxes_gt")

    system_prompt = (
        "You are a visual grounding assistant. Given an image and an "
        "instruction describing one or more target objects, reply with a "
        "single JSON object of the form "
        '{"bboxes": [[x1, y1, x2, y2], ...]} listing the bounding boxes '
        "of every matching object in absolute pixel coordinates. Do not "
        "include any text outside the JSON object."
    )

    def rewards(self):
        return [
            (json_format_reward, 0.1),
            (bbox_schema_reward, 0.1),
            (ciou_reward, 1.0),
            (hungarian_reward, 1.0),
        ]
