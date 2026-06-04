"""Grounding metric contract tests.

Pins the user-visible behavior of ``grounding_iou`` (legacy permissive)
and ``grounding_iou_f1`` (strict, reward-aligned), plus the scipy-or-greedy
fallback for Hungarian matching. Internal-helper edge cases live below
the surface — only regressions Codex actually caught are pinned here.
"""

from __future__ import annotations

import sys

import pytest

pytestmark = pytest.mark.data


class TestGroundingIouF1:
    """Multi-bbox F1 metric — the new contract used by mgrounding_test."""

    def test_empty_pred_empty_gt_is_one(self):
        from leap_finetune.evaluation.metrics import score_grounding_iou_f1

        # Correct abstention: model emits no boxes when GT has no boxes.
        assert score_grounding_iou_f1("[]", "[]") == 1.0

    def test_empty_pred_nonempty_gt_is_zero(self):
        from leap_finetune.evaluation.metrics import score_grounding_iou_f1

        assert score_grounding_iou_f1("[]", '[{"label":"x","bbox":[0,0,1,1]}]') == 0.0

    def test_multi_permuted_still_matches(self):
        """Hungarian matching is order-invariant."""
        from leap_finetune.evaluation.metrics import score_grounding_iou_f1

        s = score_grounding_iou_f1(
            '[{"label":"b","bbox":[0.5,0.5,1,1]},{"label":"a","bbox":[0,0,0.5,0.5]}]',
            '[{"label":"a","bbox":[0,0,0.5,0.5]},{"label":"b","bbox":[0.5,0.5,1,1]}]',
        )
        assert s == pytest.approx(1.0)

    def test_extra_pred_drags_precision(self):
        """2 preds vs 1 gt, one matches perfectly → F1=2/3."""
        from leap_finetune.evaluation.metrics import score_grounding_iou_f1

        s = score_grounding_iou_f1(
            '[{"label":"a","bbox":[0,0,0.5,0.5]},{"label":"b","bbox":[0.6,0.6,1,1]}]',
            '[{"label":"a","bbox":[0,0,0.5,0.5]}]',
        )
        assert s == pytest.approx(2 / 3, abs=1e-6)

    def test_registered_in_dispatch(self):
        from leap_finetune.evaluation.metrics import compute_metric

        s = compute_metric(
            "grounding_iou_f1",
            prediction='[{"label":"x","bbox":[0,0,1,1]}]',
            ground_truth='[{"label":"x","bbox":[0,0,1,1]}]',
        )
        assert s == pytest.approx(1.0)


class TestGroundingIouLegacyFormats:
    """Permissive parser kept for refcoco-style baselines. A previous
    refactor silently broke these by routing through the strict parser —
    pins the formats that must keep working.
    """

    def test_accepts_bare_4_list(self):
        from leap_finetune.evaluation.metrics import score_grounding_iou

        assert score_grounding_iou("[0, 0, 1, 1]", "[0, 0, 1, 1]") == 1.0

    def test_accepts_prose_embedded_json(self):
        from leap_finetune.evaluation.metrics import score_grounding_iou

        assert (
            score_grounding_iou("Sure! The bbox is [0, 0, 1, 1].", "[0, 0, 1, 1]")
            == 1.0
        )

    def test_rescales_0_1000_coords(self):
        """MGrounding-native 0-1000 coord space auto-scales to 0-1."""
        from leap_finetune.evaluation.metrics import score_grounding_iou

        assert score_grounding_iou("[0, 0, 1000, 1000]", "[0, 0, 1, 1]") == 1.0


class TestGroundingIouMalformedDoesNotInflate:
    """``Benchmark.evaluate`` excludes per-sample failures from the count,
    so any parser exception silently inflates the mean. Pin the regression
    paths Codex caught.
    """

    def test_boolean_coords_rejected_not_coerced(self):
        """``[false, false, true, true]`` must NOT coerce to ``[0,0,1,1]``.

        ``bool`` is a subclass of ``int``, so float() would turn booleans
        into 0.0/1.0 and a JSON-bool prediction would score IoU=1.0 against
        a full-image GT — a free perfect score from gibberish output.
        """
        from leap_finetune.evaluation.metrics import _parse_bbox, score_grounding_iou

        assert _parse_bbox("[false, false, true, true]") is None
        assert score_grounding_iou("[false, false, true, true]", "[0, 0, 1, 1]") == 0.0

    def test_garbage_input_never_raises(self):
        """Pathological strings must return None, never raise — a raised
        exception would drop the sample from the count and inflate."""
        from leap_finetune.evaluation.metrics import _parse_bbox

        for junk in [
            "",
            '{"bbox": 42}',  # int bbox; used to crash on len(int)
            '{"bbox": "hello"}',
            '{"bbox": null}',
            '[{"bbox": 42}]',
            "[0, 0, NaN, 1]",  # NaN coord
        ]:
            assert _parse_bbox(junk) is None, f"{junk!r} should parse to None"


class TestHungarianFallback:
    """The scipy-or-greedy fallback is an environment contract: customers
    on minimal installs without scipy must still get correct multi-bbox
    matching."""

    def test_greedy_fallback_when_scipy_missing(self, monkeypatch):
        import builtins

        from leap_finetune.evaluation.metrics import _hungarian_match_iou

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "scipy" or name.startswith("scipy."):
                raise ImportError("scipy disabled for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        for mod in list(sys.modules):
            if mod.startswith("scipy"):
                monkeypatch.delitem(sys.modules, mod, raising=False)

        pred = [[0, 0, 1, 1], [0, 0, 0.5, 0.5]]
        gt = [[0, 0, 0.5, 0.5], [0, 0, 1, 1]]
        ious = _hungarian_match_iou(pred, gt)
        assert len(ious) == 2
        assert sum(ious) == pytest.approx(2.0)
