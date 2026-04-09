"""Tests for the GRPO reward-spec loader.

Run with: `uv run pytest tests/test_rewards_loader.py -v`
Or with CLI flag: `uv run pytest --configs tests/test_rewards_loader.py`
"""

import pathlib
import textwrap

import pytest

from leap_finetune.rewards import resolve_reward_specs
from leap_finetune.utils.constants import LEAP_FINETUNE_DIR

pytestmark = pytest.mark.configs


REPO_ROOT = LEAP_FINETUNE_DIR
SHIPPED_REWARDS = REPO_ROOT / "rewards"


class TestResolveRewardSpecs:
    # === Empty / None ===

    def test_none_returns_empty(self, tmp_path):
        funcs, weights = resolve_reward_specs(None, tmp_path)
        assert funcs == []
        assert weights is None

    def test_empty_list_returns_empty(self, tmp_path):
        funcs, weights = resolve_reward_specs([], tmp_path)
        assert funcs == []
        assert weights is None

    # === List of specs (no weights) ===

    def test_list_of_shipped_rewards(self, tmp_path):
        funcs, weights = resolve_reward_specs(
            [
                "./rewards/length.py::length_reward",
                "./rewards/grounding_format.py::grounding_format_reward",
            ],
            tmp_path,  # any dir — paths resolve against CWD first
        )
        assert len(funcs) == 2
        assert [f.__name__ for f in funcs] == ["length_reward", "grounding_format_reward"]
        assert weights is None

    def test_shipped_reward_is_callable_with_completions(self, tmp_path):
        funcs, _ = resolve_reward_specs(
            ["./rewards/length.py::length_reward"], tmp_path
        )
        out = funcs[0]([[{"role": "assistant", "content": "x" * 200}]])
        assert out == [1.0]

    # === Dict form with weights ===

    def test_dict_form_with_weights(self, tmp_path):
        funcs, weights = resolve_reward_specs(
            {
                "funcs": [
                    "./rewards/length.py::length_reward",
                    "./rewards/grounding_format.py::grounding_format_reward",
                ],
                "weights": [0.8, 0.2],
            },
            tmp_path,
        )
        assert len(funcs) == 2
        assert weights == [0.8, 0.2]

    def test_dict_form_without_weights(self, tmp_path):
        funcs, weights = resolve_reward_specs(
            {"funcs": ["./rewards/length.py::length_reward"]}, tmp_path
        )
        assert len(funcs) == 1
        assert weights is None

    # === Path resolution ===

    def test_absolute_path(self, tmp_path):
        abs_path = str(SHIPPED_REWARDS / "length.py")
        funcs, _ = resolve_reward_specs(
            [f"{abs_path}::length_reward"], tmp_path
        )
        assert len(funcs) == 1

    def test_customer_file_in_tmp_dir(self, tmp_path):
        """Customer drops a Python file next to their YAML — config_dir fallback finds it."""
        custom = tmp_path / "my_reward.py"
        custom.write_text(
            textwrap.dedent(
                """
                def my_reward(completions, **kwargs):
                    return [42.0] * len(completions)
                """
            )
        )
        funcs, _ = resolve_reward_specs(
            ["./my_reward.py::my_reward"], tmp_path
        )
        assert len(funcs) == 1
        assert funcs[0]([["x"], ["y"]]) == [42.0, 42.0]

    # === Error cases ===

    def test_missing_separator_raises(self, tmp_path):
        with pytest.raises(ValueError, match="malformed"):
            resolve_reward_specs(["./rewards/length.py"], tmp_path)

    def test_empty_fn_name_raises(self, tmp_path):
        with pytest.raises(ValueError, match="non-empty"):
            resolve_reward_specs(["./rewards/length.py::"], tmp_path)

    def test_empty_path_raises(self, tmp_path):
        with pytest.raises(ValueError, match="non-empty"):
            resolve_reward_specs(["::length_reward"], tmp_path)

    def test_missing_file_raises_with_candidates(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            resolve_reward_specs(
                ["./rewards/does_not_exist.py::fn"], tmp_path
            )

    def test_missing_function_raises_listing_available(self, tmp_path):
        with pytest.raises(ValueError, match="length_reward"):
            resolve_reward_specs(
                ["./rewards/length.py::not_a_real_function"], tmp_path
            )

    def test_weights_length_mismatch_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Lengths must match"):
            resolve_reward_specs(
                {
                    "funcs": ["./rewards/length.py::length_reward"],
                    "weights": [1.0, 2.0],
                },
                tmp_path,
            )

    def test_non_numeric_weight_raises(self, tmp_path):
        with pytest.raises(ValueError, match="list of numbers"):
            resolve_reward_specs(
                {
                    "funcs": ["./rewards/length.py::length_reward"],
                    "weights": ["not-a-number"],
                },
                tmp_path,
            )

    def test_invalid_shape_raises(self, tmp_path):
        with pytest.raises(ValueError, match="must be a list or a dict"):
            resolve_reward_specs("not-a-list-or-dict", tmp_path)

    def test_non_string_spec_raises(self, tmp_path):
        with pytest.raises(ValueError, match="must be a string"):
            resolve_reward_specs([42], tmp_path)


class TestRecipes:
    """Tests for the Recipe class mechanism: rewards.recipe: <path>::<ClassName>."""

    def test_shipped_recipe_uses_default_weights(self, tmp_path):
        funcs, weights = resolve_reward_specs(
            {"recipe": "./rewards/vlm_grounding.py::VLMGroundingRecipe"}, tmp_path
        )
        assert len(funcs) == 4
        assert weights == [0.1, 0.1, 1.0, 1.0]
        assert [f.__name__ for f in funcs] == [
            "json_format_reward",
            "bbox_schema_reward",
            "ciou_reward",
            "hungarian_reward",
        ]

    def test_recipe_plus_individual_funcs_stacks(self, tmp_path):
        funcs, weights = resolve_reward_specs(
            {
                "recipe": "./rewards/vlm_grounding.py::VLMGroundingRecipe",
                "funcs": ["./rewards/length.py::length_reward"],
            },
            tmp_path,
        )
        assert len(funcs) == 5
        # Recipe weights first, then 1.0 for each stacked individual reward.
        assert weights == [0.1, 0.1, 1.0, 1.0, 1.0]
        assert funcs[-1].__name__ == "length_reward"

    def test_recipe_weights_override(self, tmp_path):
        _, weights = resolve_reward_specs(
            {
                "recipe": "./rewards/vlm_grounding.py::VLMGroundingRecipe",
                "weights": [0.2, 0.2, 2.0, 2.0],
            },
            tmp_path,
        )
        assert weights == [0.2, 0.2, 2.0, 2.0]

    def test_recipe_and_funcs_combined_weights_override(self, tmp_path):
        funcs, weights = resolve_reward_specs(
            {
                "recipe": "./rewards/vlm_grounding.py::VLMGroundingRecipe",
                "funcs": ["./rewards/length.py::length_reward"],
                "weights": [0.1, 0.1, 1.0, 1.0, 0.5],
            },
            tmp_path,
        )
        assert len(funcs) == 5
        assert weights == [0.1, 0.1, 1.0, 1.0, 0.5]

    def test_recipe_weights_length_mismatch_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Lengths must match"):
            resolve_reward_specs(
                {
                    "recipe": "./rewards/vlm_grounding.py::VLMGroundingRecipe",
                    "weights": [1.0],
                },
                tmp_path,
            )

    def test_missing_class_name_raises_with_alternatives(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            resolve_reward_specs(
                {"recipe": "./rewards/vlm_grounding.py::NotARecipe"}, tmp_path
            )

    def test_malformed_recipe_spec_raises(self, tmp_path):
        with pytest.raises(ValueError, match="malformed"):
            resolve_reward_specs(
                {"recipe": "./rewards/vlm_grounding.py"},  # missing ::ClassName
                tmp_path,
            )

    def test_non_recipe_class_rejected(self, tmp_path):
        not_a_recipe = tmp_path / "not_a_recipe.py"
        not_a_recipe.write_text("class NotARecipe:\n    pass\n")
        with pytest.raises(ValueError, match="subclass leap_finetune.rewards.Recipe"):
            resolve_reward_specs(
                {"recipe": f"{not_a_recipe}::NotARecipe"}, tmp_path
            )

    def test_empty_rewards_list_rejected(self, tmp_path):
        empty = tmp_path / "empty_recipe.py"
        empty.write_text(
            "from leap_finetune.rewards import Recipe\n"
            "class EmptyRecipe(Recipe):\n"
            "    def rewards(self):\n"
            "        return []\n"
        )
        with pytest.raises(ValueError, match="empty list"):
            resolve_reward_specs({"recipe": f"{empty}::EmptyRecipe"}, tmp_path)

    def test_malformed_rewards_tuple_rejected(self, tmp_path):
        """rewards() must return list of (callable, float) tuples."""
        wrong = tmp_path / "wrong_shape.py"
        wrong.write_text(
            "from leap_finetune.rewards import Recipe\n"
            "def f(completions, **kw):\n"
            "    return [1.0] * len(completions)\n"
            "class WrongRecipe(Recipe):\n"
            "    def rewards(self):\n"
            "        return [f]  # forgot the weight\n"
        )
        with pytest.raises(ValueError, match="tuple"):
            resolve_reward_specs({"recipe": f"{wrong}::WrongRecipe"}, tmp_path)

    def test_customer_extension_via_load_recipe(self, tmp_path):
        """The documented extension pattern: customer file imports a shipped
        recipe via load_recipe() and subclasses it."""
        import pathlib

        shipped = pathlib.Path("rewards/vlm_grounding.py").resolve()
        custom = tmp_path / "my_recipe.py"
        custom.write_text(
            "from leap_finetune.rewards import load_recipe\n"
            f"BaseRecipe = load_recipe(r'{shipped}::VLMGroundingRecipe')\n"
            "\n"
            "def my_extra_reward(completions, **kwargs):\n"
            "    return [0.7] * len(completions)\n"
            "\n"
            "class MyGroundingRecipe(BaseRecipe):\n"
            "    required_columns = BaseRecipe.required_columns + ('extra_col',)\n"
            "    def rewards(self):\n"
            "        return [*super().rewards(), (my_extra_reward, 0.5)]\n"
        )
        funcs, weights = resolve_reward_specs(
            {"recipe": f"{custom}::MyGroundingRecipe"}, tmp_path
        )
        assert len(funcs) == 5
        assert weights == [0.1, 0.1, 1.0, 1.0, 0.5]
        assert funcs[-1].__name__ == "my_extra_reward"

    def test_customer_subclass_preserves_parent_required_columns(self, tmp_path):
        """Extension should be able to read and extend parent class attributes."""
        import pathlib

        shipped = pathlib.Path("rewards/vlm_grounding.py").resolve()
        custom = tmp_path / "my_recipe.py"
        custom.write_text(
            "from leap_finetune.rewards import load_recipe\n"
            f"BaseRecipe = load_recipe(r'{shipped}::VLMGroundingRecipe')\n"
            "\n"
            "class ExtendedRecipe(BaseRecipe):\n"
            "    required_columns = BaseRecipe.required_columns + ('extra_col',)\n"
            "    def rewards(self):\n"
            "        return super().rewards()\n"
        )
        # We can't test required_columns through the loader (it doesn't
        # expose it yet), but we can at least verify the file imports cleanly
        # and the loader accepts the subclass.
        funcs, _ = resolve_reward_specs(
            {"recipe": f"{custom}::ExtendedRecipe"}, tmp_path
        )
        assert len(funcs) == 4
