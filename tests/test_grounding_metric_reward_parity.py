"""Locks in parser/score parity between the grounding eval metrics and the
GRPO grounding reward.

The eval metric (``leap_finetune.evaluation.metrics``) and the GRPO reward
(``rewards/tasks/vlm_grounding/recipe.py``) duplicate strict bbox parsing
rules — preamble-free strict ``json.loads``, list of ``{"bbox": [...]}``
dicts only, coords in ``[0, 1]``, ``x2>x1 and y2>y1``, no NaN/Inf, no
corner-swap. A future edit to either parser that loosens or tightens
either side independently is a real correctness bug: the model would
score differently under train signal vs eval. These tests catch that.

Run: ``uv run pytest tests/test_grounding_metric_reward_parity.py -v``
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

# ``score_grounding_iou_f1`` lands on main with the feature/async-eval PR.
# Until that merges, this branch only has ``score_grounding_iou`` — skip the
# whole module so PR A is pytest-clean standalone. Auto-re-enables on
# rebase once the parity target exists.
try:
    from leap_finetune.evaluation.metrics import (
        score_grounding_iou,
        score_grounding_iou_f1,
    )
except ImportError:
    pytest.skip(
        "score_grounding_iou_f1 not yet on this branch; "
        "merge feature/async-eval first then rebase to re-enable",
        allow_module_level=True,
    )

pytestmark = pytest.mark.data

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def reward_module():
    """Load the GRPO grounding recipe by path (it lives outside src/)."""
    spec = importlib.util.spec_from_file_location(
        "grounding_recipe", _REPO_ROOT / "rewards/tasks/vlm_grounding/recipe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reward_f1(reward_module, prediction: str, ground_truth: str) -> float:
    return reward_module.iou_f1_reward(
        [[{"content": prediction}]], solution=[ground_truth]
    )[0]


def _J(items: list) -> str:
    return json.dumps(items, ensure_ascii=False)


_GT_SINGLE = _J([{"label": "x", "bbox": [0.1, 0.1, 0.5, 0.5]}])
_GT_TRACKING_5 = _J(
    [
        {"label": "tracked object", "bbox": [i * 0.1, i * 0.1, i * 0.1 + 0.4, i * 0.1 + 0.4]}
        for i in range(5)
    ]
)


# ---------------------------------------------------------------------------
# Parity: every case maps the same input string to the same score in eval and
# reward. Any future edit that diverges them fails one of these IDs.
# ---------------------------------------------------------------------------

_CASES_VALID = [
    pytest.param(_GT_SINGLE, _GT_SINGLE, id="single-perfect"),
    pytest.param(_GT_TRACKING_5, _GT_TRACKING_5, id="tracking-5-perfect"),
    pytest.param(
        _J(list(reversed(json.loads(_GT_TRACKING_5)))), _GT_TRACKING_5,
        id="tracking-5-reversed-hungarian",
    ),
    pytest.param(
        _J(json.loads(_GT_TRACKING_5)[:3]), _GT_TRACKING_5, id="tracking-3-of-5",
    ),
    pytest.param(
        _J([{"label": "x", "bbox": [0.12, 0.12, 0.5, 0.5]}]), _GT_SINGLE,
        id="single-slightly-off",
    ),
    pytest.param(
        _J([{"label": "x", "bbox": [0.0, 0.0, 1.0, 1.0]}]), _GT_SINGLE,
        id="full-image-bbox-legitimate",
    ),
    pytest.param("[]", "[]", id="abstention"),
]

_CASES_REJECT_TO_ZERO = [
    pytest.param("[]", _GT_SINGLE, id="empty-pred-vs-valid-gt"),
    pytest.param(
        _J([{"label": "x", "bbox": [0.5, 0.5, 0.1, 0.1]}]), _GT_SINGLE,
        id="reject-inverted-corners",
    ),
    pytest.param(
        _J([{"label": "x", "bbox": [0.3, 0.3, 0.3, 0.3]}]), _GT_SINGLE,
        id="reject-zero-area",
    ),
    pytest.param(
        _J([{"label": "x", "bbox": [float("nan"), 0.1, 0.5, 0.5]}]), _GT_SINGLE,
        id="reject-nan-coord",
    ),
    pytest.param(
        _J([{"label": "x", "bbox": [float("inf"), 0.1, 0.5, 0.5]}]), _GT_SINGLE,
        id="reject-inf-coord",
    ),
    pytest.param(
        _J([{"label": "x", "bbox": [0.1, 0.1, 1.5, 1.5]}]), _GT_SINGLE,
        id="reject-coord-above-one",
    ),
    pytest.param(
        _J([{"label": "x", "bbox": [-0.4, -0.4, 0.5, 0.5]}]), _GT_SINGLE,
        id="reject-negative-coord",
    ),
    pytest.param(
        _J([{"label": "x", "bbox": [100, 100, 500, 500]}]), _GT_SINGLE,
        id="reject-0-1000-coords",
    ),
    pytest.param("Sure: " + _GT_SINGLE, _GT_SINGLE, id="reject-preamble-before-json"),
    pytest.param(
        "[{'bbox':[0.1,0.1,0.5,0.5]}]", _GT_SINGLE,
        id="reject-python-single-quotes",
    ),
    pytest.param("[0.1,0.1,0.5,0.5]", _GT_SINGLE, id="reject-bare-flat-list"),
    pytest.param("[[0.1,0.1,0.5,0.5]]", _GT_SINGLE, id="reject-list-of-lists"),
    pytest.param("absolutely not json", _GT_SINGLE, id="reject-garbage-text"),
    pytest.param(
        '[{"bbox": [0.1,0.1,0.5,0.5],}]', _GT_SINGLE,
        id="reject-trailing-comma-json",
    ),
]


@pytest.mark.parametrize("pred,gt", _CASES_VALID + _CASES_REJECT_TO_ZERO)
def test_eval_f1_matches_reward_f1(reward_module, pred, gt):
    """``grounding_iou_f1`` (eval) returns the same score as ``iou_f1_reward``
    (GRPO reward). Catches any future drift in either parser.
    """
    eval_score = score_grounding_iou_f1(pred, gt)
    reward_score = _reward_f1(reward_module, pred, gt)
    assert eval_score == pytest.approx(reward_score, abs=1e-9), (
        f"eval {eval_score} != reward {reward_score} for pred={pred!r} gt={gt!r}"
    )


@pytest.mark.parametrize("pred,gt", _CASES_REJECT_TO_ZERO)
def test_malformed_outputs_score_zero(reward_module, pred, gt):
    """All malformed/out-of-range/format-bypass attempts must score exactly
    0.0 on both paths — no silent partial credit from accidental overlap.
    """
    assert score_grounding_iou_f1(pred, gt) == 0.0
    assert _reward_f1(reward_module, pred, gt) == 0.0


# ---------------------------------------------------------------------------
# The single-bbox refcoco metric ``grounding_iou`` is still thresholded at
# IoU=0.5 (accuracy@.5 is the canonical refcoco metric) but MUST use the same
# strict parser — its previous incarnation accepted preambles, bare lists,
# 0-1000 coords, and corner-swapped inverted boxes, all of which silently
# inflated the scores reported in the cookbook.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pred,gt", _CASES_REJECT_TO_ZERO)
def test_single_bbox_metric_rejects_malformed(pred, gt):
    """``score_grounding_iou`` must score 0 on every malformed input, same as
    the multi-bbox path. Catches a regression to the old lenient parser.
    """
    assert score_grounding_iou(pred, gt) == 0.0


def test_single_bbox_metric_preserves_threshold():
    """The thresholded 0/1 semantic is preserved on legitimate predictions."""
    gt = _GT_SINGLE
    # ~0.86 IoU — above threshold → 1.0
    above = _J([{"label": "x", "bbox": [0.12, 0.12, 0.5, 0.5]}])
    # ~0.19 IoU — below threshold → 0.0
    below = _J([{"label": "x", "bbox": [0.3, 0.3, 0.6, 0.6]}])
    assert score_grounding_iou(above, gt) == 1.0
    assert score_grounding_iou(below, gt) == 0.0


# ---------------------------------------------------------------------------
# Numeric value of F1 in non-trivial cases — pins the math so refactors that
# alter scoring (rather than just parsing) also fail visibly.
# ---------------------------------------------------------------------------

def test_tracking_three_of_five_f1_value():
    """Three perfect bboxes covering only the first three of five GT frames
    yields P=1.0, R=3/5, F1=2·1·0.6/(1+0.6)=0.75 exactly.
    """
    pred = _J(json.loads(_GT_TRACKING_5)[:3])
    assert score_grounding_iou_f1(pred, _GT_TRACKING_5) == pytest.approx(0.75, abs=1e-9)
