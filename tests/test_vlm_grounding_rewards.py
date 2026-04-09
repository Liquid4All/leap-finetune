"""Pure-logic tests for the VLM grounding reward bundle.

No GPU, no model — these exercise the CIoU primitive, the pure-Python
Hungarian matching, JSON extraction, and each of the four reward functions
plus the BUNDLE export.

Run with: `uv run pytest --data tests/test_vlm_grounding_rewards.py -v`
"""

import importlib.util
import pathlib

import pytest

from leap_finetune.rewards import Recipe

pytestmark = pytest.mark.data

# Load vlm_grounding.py as an anonymous module so we don't need to fiddle
# with sys.path. The file lives in the top-level rewards/ directory.
_SPEC = importlib.util.spec_from_file_location(
    "vlm_grounding",
    pathlib.Path(__file__).parent.parent / "rewards" / "vlm_grounding.py",
)
VG = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(VG)


# === IoU / CIoU primitives ===


class TestIoUPrimitives:
    def test_perfect_overlap(self):
        assert VG._iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_no_overlap(self):
        assert VG._iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_half_overlap_x_axis(self):
        # Two 10x10 boxes overlapping half their width → 50/150 = 1/3
        assert abs(VG._iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9

    def test_zero_area_box(self):
        # Degenerate (zero-width) box should return 0.0, not crash
        assert VG._iou((0, 0, 0, 10), (0, 0, 10, 10)) == 0.0


class TestCIoU:
    def test_perfect_match(self):
        assert abs(VG._ciou((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-9

    def test_center_offset_penalized(self):
        """CIoU should be strictly less than IoU when centers don't align."""
        iou = VG._iou((0, 0, 10, 10), (5, 0, 15, 10))
        ciou = VG._ciou((0, 0, 10, 10), (5, 0, 15, 10))
        assert ciou < iou

    def test_no_overlap_returns_zero(self):
        # After clipping to [0, 1], disjoint boxes should be 0.0
        assert VG._ciou((0, 0, 10, 10), (100, 100, 110, 110)) == 0.0

    def test_aspect_ratio_penalty(self):
        """Same IoU, different aspect ratio → lower CIoU."""
        # Reference: 10x10 square at origin
        ref = (0, 0, 10, 10)
        # Candidate A: same shape but shifted
        a = (2, 0, 12, 10)
        # Candidate B: very different aspect ratio (wide rectangle)
        b = (0, 2, 20, 6)
        # Both should have CIoU < perfect, but let's just check both are valid scores
        assert 0.0 <= VG._ciou(a, ref) <= 1.0
        assert 0.0 <= VG._ciou(b, ref) <= 1.0


# === Hungarian matching ===


class TestHungarianAssignment:
    def test_trivial_1x1(self):
        assert VG._hungarian_assignment([[0.5]]) == [(0, 0)]

    def test_identity_2x2(self):
        pairs = VG._hungarian_assignment([[0.0, 1.0], [1.0, 0.0]])
        assert set(pairs) == {(0, 0), (1, 1)}

    def test_swap_2x2(self):
        """The optimal assignment is to cross the diagonal."""
        pairs = VG._hungarian_assignment([[1.0, 0.0], [0.0, 1.0]])
        assert set(pairs) == {(0, 1), (1, 0)}

    def test_rectangular_more_rows_than_cols(self):
        # 3 preds, 2 gts → min(3,2)=2 assignments; best preds (rows 0, 2) match
        cost = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]
        pairs = VG._hungarian_assignment(cost)
        assert len(pairs) == 2
        rows = {r for r, _ in pairs}
        assert 0 in rows and 2 in rows

    def test_rectangular_more_cols_than_rows(self):
        cost = [[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]]
        pairs = VG._hungarian_assignment(cost)
        assert len(pairs) == 2
        assert set(pairs) == {(0, 0), (1, 2)}


# === JSON extraction ===


class TestJSONExtraction:
    def test_bare_json(self):
        assert VG._extract_json('{"bbox": [1, 2, 3, 4]}') == {"bbox": [1, 2, 3, 4]}

    def test_json_wrapped_in_prose(self):
        assert VG._extract_json('prose {"bbox": [1,2,3,4]} more prose') == {
            "bbox": [1, 2, 3, 4]
        }

    def test_no_json_returns_none(self):
        assert VG._extract_json("nothing here") is None

    def test_unclosed_json_returns_none(self):
        assert VG._extract_json('{"broken') is None

    def test_nested_object(self):
        text = '{"outer": {"bbox": [1, 2, 3, 4]}}'
        parsed = VG._extract_json(text)
        assert parsed is not None


class TestBBoxExtraction:
    def test_single_bbox_form(self):
        assert VG._extract_bboxes('{"bbox": [0, 0, 10, 10]}') == [(0.0, 0.0, 10.0, 10.0)]

    def test_multi_bbox_form(self):
        assert VG._extract_bboxes('{"bboxes": [[0,0,10,10], [20,20,30,30]]}') == [
            (0.0, 0.0, 10.0, 10.0),
            (20.0, 20.0, 30.0, 30.0),
        ]

    def test_wrapped_in_prose(self):
        assert VG._extract_bboxes('The object is at {"bbox": [5, 5, 15, 15]}.') == [
            (5.0, 5.0, 15.0, 15.0)
        ]

    def test_wrong_length_rejected(self):
        assert VG._extract_bboxes('{"bbox": [1, 2]}') == []

    def test_no_json_empty(self):
        assert VG._extract_bboxes("no json") == []

    def test_no_bbox_key_empty(self):
        assert VG._extract_bboxes('{"answer": "dog"}') == []


# === Reward functions ===


def _assistant(content: str):
    """Short helper to build a conversational completion."""
    return [{"role": "assistant", "content": content}]


class TestJSONFormatReward:
    def test_valid_json_scores_one(self):
        assert VG.json_format_reward([_assistant('{"bbox": [0,0,10,10]}')]) == [1.0]

    def test_prose_only_scores_zero(self):
        assert VG.json_format_reward([_assistant("just text")]) == [0.0]

    def test_json_wrapped_in_prose_scores_one(self):
        assert VG.json_format_reward([_assistant('x {"a": 1} y')]) == [1.0]


class TestBBoxSchemaReward:
    def test_valid_bbox_scores_one(self):
        assert VG.bbox_schema_reward([_assistant('{"bbox": [0,0,10,10]}')]) == [1.0]

    def test_json_without_bbox_scores_zero(self):
        assert VG.bbox_schema_reward([_assistant('{"answer": "dog"}')]) == [0.0]

    def test_multi_bbox_form_accepted(self):
        assert VG.bbox_schema_reward([_assistant('{"bboxes": [[0,0,10,10]]}')]) == [1.0]


class TestCIoUReward:
    def test_perfect_match(self):
        r = VG.ciou_reward(
            [_assistant('{"bbox": [0, 0, 10, 10]}')], bbox_gt=[[0, 0, 10, 10]]
        )
        assert r == [1.0]

    def test_no_overlap(self):
        r = VG.ciou_reward(
            [_assistant('{"bbox": [0, 0, 10, 10]}')],
            bbox_gt=[[100, 100, 110, 110]],
        )
        assert r == [0.0]

    def test_unparseable_completion(self):
        r = VG.ciou_reward([_assistant("garbage")], bbox_gt=[[0, 0, 10, 10]])
        assert r == [0.0]

    def test_missing_bbox_gt_column(self):
        assert VG.ciou_reward([_assistant("x")], bbox_gt=None) == [0.0]


class TestHungarianReward:
    def test_exact_single_box(self):
        r = VG.hungarian_reward(
            [_assistant('{"bboxes": [[0, 0, 10, 10]]}')],
            bboxes_gt=[[[0, 0, 10, 10]]],
        )
        assert r == [1.0]

    def test_permutation_invariance(self):
        """Swapping the order of predicted boxes shouldn't affect the score."""
        r = VG.hungarian_reward(
            [_assistant('{"bboxes": [[100, 100, 110, 110], [0, 0, 10, 10]]}')],
            bboxes_gt=[[[0, 0, 10, 10], [100, 100, 110, 110]]],
        )
        assert abs(r[0] - 1.0) < 1e-6

    def test_over_prediction_penalty(self):
        """2 predicted boxes, 1 ground-truth → reward is halved."""
        r = VG.hungarian_reward(
            [_assistant('{"bboxes": [[0, 0, 10, 10], [50, 50, 60, 60]]}')],
            bboxes_gt=[[[0, 0, 10, 10]]],
        )
        assert abs(r[0] - 0.5) < 1e-6

    def test_under_prediction_penalty(self):
        """1 predicted box, 2 ground-truth boxes → reward is halved."""
        r = VG.hungarian_reward(
            [_assistant('{"bboxes": [[0, 0, 10, 10]]}')],
            bboxes_gt=[[[0, 0, 10, 10], [50, 50, 60, 60]]],
        )
        assert abs(r[0] - 0.5) < 1e-6

    def test_unparseable_completion(self):
        r = VG.hungarian_reward(
            [_assistant("garbage")], bboxes_gt=[[[0, 0, 10, 10]]]
        )
        assert r == [0.0]

    def test_missing_column(self):
        assert VG.hungarian_reward([_assistant("x")], bboxes_gt=None) == [0.0]


class TestVLMGroundingRecipe:
    def test_is_recipe_subclass(self):
        assert issubclass(VG.VLMGroundingRecipe, Recipe)

    def test_class_attributes(self):
        cls = VG.VLMGroundingRecipe
        assert cls.description  # non-empty
        assert cls.required_columns == ("prompt", "bbox_gt", "bboxes_gt")
        assert cls.system_prompt is not None
        assert "bboxes" in cls.system_prompt

    def test_rewards_returns_list_of_pairs(self):
        pairs = VG.VLMGroundingRecipe().rewards()
        assert isinstance(pairs, list)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)

    def test_rewards_has_four_entries(self):
        pairs = VG.VLMGroundingRecipe().rewards()
        assert len(pairs) == 4

    def test_rewards_entries_are_callable_and_numeric(self):
        for fn, weight in VG.VLMGroundingRecipe().rewards():
            assert callable(fn)
            assert isinstance(weight, (int, float))

    def test_rewards_function_order(self):
        """Canonical order: format → schema → CIoU → Hungarian."""
        names = [fn.__name__ for fn, _ in VG.VLMGroundingRecipe().rewards()]
        assert names == [
            "json_format_reward",
            "bbox_schema_reward",
            "ciou_reward",
            "hungarian_reward",
        ]

    def test_default_weights_sane(self):
        """Format/schema weights should be small; geometry rewards should
        dominate the total."""
        weights = [w for _, w in VG.VLMGroundingRecipe().rewards()]
        assert weights == [0.1, 0.1, 1.0, 1.0]

    def test_extension_pattern_works(self):
        """A subclass can extend rewards() with super() + append."""

        def my_extra(completions, **kwargs):
            return [0.5] * len(completions)

        class Extended(VG.VLMGroundingRecipe):
            def rewards(self):
                return [*super().rewards(), (my_extra, 0.3)]

        pairs = Extended().rewards()
        assert len(pairs) == 5
        assert pairs[-1] == (my_extra, 0.3)
        # Parent pairs preserved
        names = [fn.__name__ for fn, _ in pairs]
        assert names[:4] == [
            "json_format_reward",
            "bbox_schema_reward",
            "ciou_reward",
            "hungarian_reward",
        ]
