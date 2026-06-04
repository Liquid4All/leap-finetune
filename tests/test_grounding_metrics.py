"""Unit tests for grounding metrics in evaluation/metrics.py.

Covers the JSON-bbox parser, IoU computation, Hungarian matching
(scipy path + greedy fallback), and the ``grounding_iou_f1`` scorer.
"""

from __future__ import annotations

import sys

import pytest

pytestmark = pytest.mark.data


# === Parser strictness ===


class TestParseBboxes:
    def test_rejects_prose_preamble(self):
        from leap_finetune.evaluation.metrics import _parse_bboxes

        # Strict parse: JSON must be the whole output, not prose+JSON.
        assert _parse_bboxes('Here you go: [{"label":"x","bbox":[0,0,1,1]}]') == []

    def test_rejects_bare_list_of_lists(self):
        from leap_finetune.evaluation.metrics import _parse_bboxes

        # Must be list-of-dicts with bbox key; bare coords don't count.
        assert _parse_bboxes("[[0,0,1,1]]") == []

    def test_rejects_out_of_range_coords(self):
        from leap_finetune.evaluation.metrics import _parse_bboxes

        assert _parse_bboxes('[{"label":"x","bbox":[0,0,1,1.5]}]') == []
        assert _parse_bboxes('[{"label":"x","bbox":[-0.1,0,1,1]}]') == []

    def test_rejects_zero_or_inverted_area(self):
        from leap_finetune.evaluation.metrics import _parse_bboxes

        assert _parse_bboxes('[{"label":"x","bbox":[0.5,0,0.5,1]}]') == []  # x2==x1
        assert _parse_bboxes('[{"label":"x","bbox":[1,0,0,1]}]') == []  # x2<x1

    def test_accepts_well_formed_multi_box(self):
        from leap_finetune.evaluation.metrics import _parse_bboxes

        boxes = _parse_bboxes(
            '[{"label":"a","bbox":[0,0,0.5,0.5]},{"label":"b","bbox":[0.5,0.5,1,1]}]'
        )
        assert len(boxes) == 2
        assert boxes[0] == [0.0, 0.0, 0.5, 0.5]


# === IoU + Hungarian matcher ===


class TestHungarianMatch:
    def test_picks_optimal_assignment_not_greedy(self):
        from leap_finetune.evaluation.metrics import _hungarian_match_iou

        # IoUs: A-A'=0.25, A-B'=1.0, B-A'=1.0, B-B'=0.25. Optimal matching
        # is the diagonal pair-swap (A->B', B->A') for total 2.0.
        pred = [[0, 0, 1, 1], [0, 0, 0.5, 0.5]]
        gt = [[0, 0, 0.5, 0.5], [0, 0, 1, 1]]
        ious = _hungarian_match_iou(pred, gt)
        assert sum(ious) == pytest.approx(2.0)

    def test_returns_min_length_on_mismatched_counts(self):
        from leap_finetune.evaluation.metrics import _hungarian_match_iou

        pred = [[0, 0, 1, 1]]
        gt = [[0, 0, 1, 1], [0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1]]
        ious = _hungarian_match_iou(pred, gt)
        assert len(ious) == 1
        assert ious[0] == pytest.approx(1.0)

    def test_empty_returns_empty(self):
        from leap_finetune.evaluation.metrics import _hungarian_match_iou

        assert _hungarian_match_iou([], [[0, 0, 1, 1]]) == []
        assert _hungarian_match_iou([[0, 0, 1, 1]], []) == []
        assert _hungarian_match_iou([], []) == []

    def test_greedy_fallback_when_scipy_missing(self, monkeypatch):
        """Force ImportError on scipy, verify greedy path runs and gives
        a reasonable answer on a small bipartite problem."""
        import builtins

        from leap_finetune.evaluation.metrics import _hungarian_match_iou

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "scipy" or name.startswith("scipy."):
                raise ImportError("scipy disabled for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        # Also evict any cached scipy module so the lazy import really re-runs.
        for mod in list(sys.modules):
            if mod.startswith("scipy"):
                monkeypatch.delitem(sys.modules, mod, raising=False)

        pred = [[0, 0, 1, 1], [0, 0, 0.5, 0.5]]
        gt = [[0, 0, 0.5, 0.5], [0, 0, 1, 1]]
        ious = _hungarian_match_iou(pred, gt)
        assert len(ious) == 2
        assert sum(ious) == pytest.approx(2.0)  # optimal/greedy both reach 2.0


# === grounding_iou_f1 boundary cases ===


class TestGroundingIouF1:
    def test_empty_pred_empty_gt_is_one(self):
        from leap_finetune.evaluation.metrics import score_grounding_iou_f1

        assert score_grounding_iou_f1("[]", "[]") == 1.0

    def test_empty_pred_nonempty_gt_is_zero(self):
        from leap_finetune.evaluation.metrics import score_grounding_iou_f1

        assert score_grounding_iou_f1("[]", '[{"label":"x","bbox":[0,0,1,1]}]') == 0.0

    def test_nonempty_pred_empty_gt_is_zero(self):
        from leap_finetune.evaluation.metrics import score_grounding_iou_f1

        assert score_grounding_iou_f1('[{"label":"x","bbox":[0,0,1,1]}]', "[]") == 0.0

    def test_single_perfect_match_is_one(self):
        from leap_finetune.evaluation.metrics import score_grounding_iou_f1

        s = score_grounding_iou_f1(
            '[{"label":"x","bbox":[0,0,1,1]}]',
            '[{"label":"x","bbox":[0,0,1,1]}]',
        )
        assert s == pytest.approx(1.0)

    def test_single_no_overlap_is_zero(self):
        from leap_finetune.evaluation.metrics import score_grounding_iou_f1

        s = score_grounding_iou_f1(
            '[{"label":"x","bbox":[0,0,0.4,0.4]}]',
            '[{"label":"x","bbox":[0.6,0.6,1,1]}]',
        )
        assert s == 0.0

    def test_multi_perfect_match(self):
        from leap_finetune.evaluation.metrics import score_grounding_iou_f1

        s = score_grounding_iou_f1(
            ('[{"label":"a","bbox":[0,0,0.5,0.5]},{"label":"b","bbox":[0.5,0.5,1,1]}]'),
            ('[{"label":"a","bbox":[0,0,0.5,0.5]},{"label":"b","bbox":[0.5,0.5,1,1]}]'),
        )
        assert s == pytest.approx(1.0)

    def test_multi_permuted_still_matches(self):
        """Hungarian matching is order-invariant: same boxes in different
        order should still score 1.0."""
        from leap_finetune.evaluation.metrics import score_grounding_iou_f1

        s = score_grounding_iou_f1(
            ('[{"label":"b","bbox":[0.5,0.5,1,1]},{"label":"a","bbox":[0,0,0.5,0.5]}]'),
            ('[{"label":"a","bbox":[0,0,0.5,0.5]},{"label":"b","bbox":[0.5,0.5,1,1]}]'),
        )
        assert s == pytest.approx(1.0)

    def test_extra_pred_drags_precision(self):
        """2 preds vs 1 gt, one matches perfectly. P=0.5, R=1.0, F1=2/3."""
        from leap_finetune.evaluation.metrics import score_grounding_iou_f1

        s = score_grounding_iou_f1(
            ('[{"label":"a","bbox":[0,0,0.5,0.5]},{"label":"b","bbox":[0.6,0.6,1,1]}]'),
            '[{"label":"a","bbox":[0,0,0.5,0.5]}]',
        )
        assert s == pytest.approx(2 / 3, abs=1e-6)

    def test_missing_pred_drags_recall(self):
        """1 pred vs 2 gts, one matches perfectly. P=1.0, R=0.5, F1=2/3."""
        from leap_finetune.evaluation.metrics import score_grounding_iou_f1

        s = score_grounding_iou_f1(
            '[{"label":"a","bbox":[0,0,0.5,0.5]}]',
            ('[{"label":"a","bbox":[0,0,0.5,0.5]},{"label":"b","bbox":[0.6,0.6,1,1]}]'),
        )
        assert s == pytest.approx(2 / 3, abs=1e-6)

    def test_malformed_pred_scores_zero(self):
        from leap_finetune.evaluation.metrics import score_grounding_iou_f1

        # Strict parser rejects prose-embedded JSON → empty preds → 0.0 vs non-empty gt.
        assert (
            score_grounding_iou_f1(
                'Sure! [{"label":"x"}]',
                '[{"label":"x","bbox":[0,0,1,1]}]',
            )
            == 0.0
        )

    def test_registered_in_dispatch(self):
        from leap_finetune.evaluation.metrics import compute_metric

        # End-to-end through the dispatcher.
        s = compute_metric(
            "grounding_iou_f1",
            prediction='[{"label":"x","bbox":[0,0,1,1]}]',
            ground_truth='[{"label":"x","bbox":[0,0,1,1]}]',
        )
        assert s == pytest.approx(1.0)


# === Score-grounding-iou (legacy single-box) sanity ===


class TestGroundingIouLegacy:
    """Pin the permissive behavior of the legacy ``grounding_iou`` metric.

    These formats were accepted before PR B; a previous refactor silently
    broke them by routing ``_parse_bbox`` through the strict multi-bbox
    parser. The pin ensures published-baseline scoring stays stable.
    """

    def test_perfect_match_is_one(self):
        from leap_finetune.evaluation.metrics import score_grounding_iou

        s = score_grounding_iou(
            '[{"label":"x","bbox":[0,0,1,1]}]',
            '[{"label":"x","bbox":[0,0,1,1]}]',
            iou_threshold=0.5,
        )
        assert s == 1.0

    def test_disjoint_is_zero(self):
        from leap_finetune.evaluation.metrics import score_grounding_iou

        s = score_grounding_iou(
            '[{"label":"x","bbox":[0,0,0.3,0.3]}]',
            '[{"label":"x","bbox":[0.7,0.7,1,1]}]',
            iou_threshold=0.5,
        )
        assert s == 0.0

    def test_accepts_bare_4_list(self):
        """Format ``[x,y,x,y]`` — bare coords, no dict wrapper."""
        from leap_finetune.evaluation.metrics import score_grounding_iou

        assert score_grounding_iou("[0, 0, 1, 1]", "[0, 0, 1, 1]") == 1.0

    def test_accepts_list_of_lists(self):
        """Format ``[[x,y,x,y]]`` — single bbox inside outer list."""
        from leap_finetune.evaluation.metrics import score_grounding_iou

        assert score_grounding_iou("[[0, 0, 1, 1]]", "[[0, 0, 1, 1]]") == 1.0

    def test_accepts_prose_embedded_json(self):
        """Format ``"The box is [0,0,1,1]"`` — JSON inside a sentence.

        Regex JSON-extraction; the strict parser would reject this.
        """
        from leap_finetune.evaluation.metrics import score_grounding_iou

        assert (
            score_grounding_iou(
                "Sure! The bbox is [0, 0, 1, 1].",
                "[0, 0, 1, 1]",
            )
            == 1.0
        )

    def test_accepts_dict_with_bbox(self):
        """Format ``{"bbox":[x,y,x,y]}`` — single top-level dict."""
        from leap_finetune.evaluation.metrics import score_grounding_iou

        assert (
            score_grounding_iou(
                '{"bbox":[0,0,1,1]}',
                '{"bbox":[0,0,1,1]}',
            )
            == 1.0
        )

    def test_rescales_0_1000_coords(self):
        """MGrounding-native 0-1000 coord space → 0-1 autoscale."""
        from leap_finetune.evaluation.metrics import score_grounding_iou

        # 0-1000 coords representing the full image.
        assert score_grounding_iou("[0, 0, 1000, 1000]", "[0, 0, 1, 1]") == 1.0

    def test_falls_back_to_ast_literal_eval(self):
        """Single-quoted JSON (invalid JSON, valid Python literal) still parses."""
        from leap_finetune.evaluation.metrics import score_grounding_iou

        assert (
            score_grounding_iou(
                "[{'bbox':[0,0,1,1]}]",
                "[{'bbox':[0,0,1,1]}]",
            )
            == 1.0
        )

    def test_below_threshold_is_zero(self):
        from leap_finetune.evaluation.metrics import score_grounding_iou

        # IoU = (0.5*0.5)/((0.5*0.5) + (1*1) - (0.5*0.5)) = 0.25
        assert (
            score_grounding_iou("[0, 0, 0.5, 0.5]", "[0, 0, 1, 1]", iou_threshold=0.5)
            == 0.0
        )


class TestGroundingIouMalformedDoesNotInflate:
    """Malformed predictions must score 0, not raise. ``Benchmark.evaluate``
    drops per-sample failures from the count, so any exception escaping
    the parser silently inflates the mean."""

    def test_bbox_value_is_int_returns_none(self):
        """``{"bbox": 42}`` used to raise ``TypeError`` from ``len(42)``."""
        from leap_finetune.evaluation.metrics import _parse_bbox

        assert _parse_bbox('{"bbox": 42}') is None

    def test_bbox_value_is_int_scores_zero(self):
        from leap_finetune.evaluation.metrics import score_grounding_iou

        assert (
            score_grounding_iou('{"bbox": 42}', "[0, 0, 1, 1]", iou_threshold=0.5)
            == 0.0
        )

    def test_bbox_value_is_dict_returns_none(self):
        from leap_finetune.evaluation.metrics import _parse_bbox

        assert _parse_bbox('{"bbox": {"x": 1}}') is None

    def test_bbox_value_is_string_returns_none(self):
        from leap_finetune.evaluation.metrics import _parse_bbox

        assert _parse_bbox('{"bbox": "(0,0,1,1)"}') is None

    def test_nan_rejected(self):
        from leap_finetune.evaluation.metrics import _parse_bbox

        # Python's json doesn't accept literal NaN but Python literals do.
        assert _parse_bbox("[0, 0, float('nan'), 1]") is None
        # ast.literal_eval rejects function calls; the parser falls through.
        assert _parse_bbox("[0, 0, 1, 1e500]") is None  # parses to inf

    def test_garbage_input_never_raises(self):
        from leap_finetune.evaluation.metrics import _parse_bbox

        # Pathological strings — every one must return None, never raise.
        for junk in [
            "",
            "not json",
            "{",
            "[",
            "{}",
            "[]",
            '{"bbox": null}',
            '{"bbox": [1, 2]}',  # wrong arity
            '{"bbox": [1, 2, 3, 4, 5]}',  # wrong arity
            '[{"label": "x"}]',  # no bbox field
            '{"box": [0, 0, 1, 1]}',  # wrong key
            '[{"bbox": 42}]',  # int bbox inside list-of-dicts
        ]:
            assert _parse_bbox(junk) is None, f"{junk!r} should parse to None"

    def test_boolean_coords_rejected_not_coerced(self):
        """``[false, false, true, true]`` must NOT coerce to ``[0,0,1,1]``.

        ``bool`` is a subclass of ``int``, so float() happily turns booleans
        into 0.0/1.0. If we allowed that, a model emitting JSON booleans as
        its bbox would score IoU=1.0 against a full-image GT — a free
        perfect score from semantic nonsense.
        """
        from leap_finetune.evaluation.metrics import _parse_bbox, score_grounding_iou

        assert _parse_bbox("[false, false, true, true]") is None
        assert _parse_bbox('[{"label":"x","bbox":[false,false,true,true]}]') is None
        assert score_grounding_iou("[false, false, true, true]", "[0, 0, 1, 1]") == 0.0
        assert (
            score_grounding_iou(
                '[{"label":"x","bbox":[false,false,true,true]}]',
                "[0, 0, 1, 1]",
            )
            == 0.0
        )
