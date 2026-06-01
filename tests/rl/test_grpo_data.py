import importlib.util
import pathlib

import pytest
import torch
from datasets import Dataset

from leap_finetune.data_loading.validate_dataset_format import (
    get_row_filter,
    normalize_columns,
    validate_dataset_format,
)

pytestmark = pytest.mark.rl


# === GRPO dataset validation ===


# === Row filter: grpo (text) ===


class TestGRPORowFilter:
    def setup_method(self):
        self.f = get_row_filter("grpo")

    def test_string_prompt_accepted(self):
        assert self.f({"prompt": "What is 2+2?"}) is True

    def test_messages_prompt_accepted(self):
        assert self.f({"prompt": [{"role": "user", "content": "hi"}]}) is True

    def test_empty_string_rejected(self):
        assert self.f({"prompt": ""}) is False
        assert self.f({"prompt": "   "}) is False

    def test_empty_list_rejected(self):
        assert self.f({"prompt": []}) is False

    def test_malformed_message_rejected(self):
        assert self.f({"prompt": [{"bad": "shape"}]}) is False

    def test_missing_column_rejected(self):
        assert self.f({}) is False

    def test_none_rejected(self):
        assert self.f({"prompt": None}) is False


# === Row filter: vlm_grpo ===


class TestVLMGRPORowFilter:
    def setup_method(self):
        # Patch is_image_loadable to always return True so we don't need
        # real files on disk for schema tests.
        from leap_finetune.data_loading import image_loader

        self._orig = image_loader.is_image_loadable
        image_loader.is_image_loadable = lambda s: True

        # The filter closes over the import at call time; reset its resolved fn
        self.f = get_row_filter("vlm_grpo")

    def teardown_method(self):
        from leap_finetune.data_loading import image_loader

        image_loader.is_image_loadable = self._orig

    def test_good_vlm_prompt_accepted(self):
        row = {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "/path.jpg"},
                        {"type": "text", "text": "describe"},
                    ],
                }
            ]
        }
        assert self.f(row) is True

    def test_string_prompt_rejected_for_vlm(self):
        assert self.f({"prompt": "just text"}) is False

    def test_unknown_content_type_rejected(self):
        row = {
            "prompt": [
                {
                    "role": "user",
                    "content": [{"type": "audio", "audio": "/x.wav"}],
                }
            ]
        }
        assert self.f(row) is False

    def test_non_string_image_rejected(self):
        row = {
            "prompt": [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": 12345}],
                }
            ]
        }
        assert self.f(row) is False


# === normalize_columns ===


class TestGRPONormalizeColumns:
    def test_grpo_aliases_rename_to_prompt(self):
        n = normalize_columns("grpo")
        assert n({"question": "hi"})["prompt"] == "hi"
        assert n({"query": "hi"})["prompt"] == "hi"
        assert n({"input": "hi"})["prompt"] == "hi"

    def test_grpo_passthrough_when_prompt_exists(self):
        n = normalize_columns("grpo")
        assert n({"prompt": "hi"})["prompt"] == "hi"

    def test_grpo_sft_messages_split(self):
        """Text GRPO normalizer splits an SFT `messages` column into
        `prompt` (non-assistant turns) and `solution` (last assistant)."""
        n = normalize_columns("grpo")
        row = {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "The answer is 4."},
            ]
        }
        result = n(row)
        assert len(result["prompt"]) == 1
        assert result["prompt"][0]["role"] == "user"
        assert result["solution"] == "The answer is 4."
        assert "messages" not in result

    def test_grpo_existing_solution_not_overwritten(self):
        """Pre-existing `solution` column takes precedence over auto-split
        so customers can keep full CoT as SFT target and a clean answer
        in `solution` for GRPO rewards."""
        n = normalize_columns("grpo")
        row = {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "Step-by-step: ... #### 4"},
            ],
            "solution": "4",
        }
        result = n(row)
        assert result["solution"] == "4"

    def test_grpo_system_message_preserved(self):
        n = normalize_columns("grpo")
        row = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        }
        result = n(row)
        assert [m["role"] for m in result["prompt"]] == ["system", "user"]
        assert result["solution"] == "hello"

    def test_grpo_ndarray_messages_converted(self):
        """Parquet-loaded list columns come back as numpy ndarrays."""
        import numpy as np

        n = normalize_columns("grpo")
        messages = np.array(
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
            dtype=object,
        )
        row = {"messages": messages}
        result = n(row)
        assert isinstance(result["prompt"], list)
        assert result["solution"] == "a"

    def test_grpo_user_only_messages_no_solution(self):
        """GRPO-only datasets (e.g. IFEval) have no assistant turn; the
        customer supplies `solution` separately and the normalizer should
        not invent one."""
        n = normalize_columns("grpo")
        row = {
            "messages": [{"role": "user", "content": "q"}],
            "solution": "preset",
        }
        result = n(row)
        assert len(result["prompt"]) == 1
        assert result["solution"] == "preset"

    def test_vlm_grpo_alias_messages_to_prompt(self):
        """VLM GRPO should be able to reuse datasets built for VLM SFT."""
        n = normalize_columns("vlm_grpo")
        row = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hi"}],
                }
            ]
        }
        result = n(row)
        assert "prompt" in result
        assert "messages" not in result

    def test_vlm_grpo_prepends_image_root(self):
        n = normalize_columns("vlm_grpo", image_root="/data")
        row = {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "subdir/pic.jpg"},
                        {"type": "image", "image": "/abs/path.jpg"},
                        {"type": "text", "text": "hi"},
                    ],
                }
            ]
        }
        content = n(row)["prompt"][0]["content"]
        assert content[0]["image"] == "/data/subdir/pic.jpg"
        assert content[1]["image"] == "/abs/path.jpg"  # absolute untouched


# === validate_dataset_format ===


class TestValidateGRPODatasetFormat:
    def test_valid_grpo_string_prompt_passes(self):
        ds = Dataset.from_list(
            [
                {"prompt": "q1", "ground_truth": "a1"},
                {"prompt": "q2", "solution": "a2"},
            ]
        )
        out = validate_dataset_format(ds, "grpo")
        assert out is ds

    def test_missing_prompt_column_raises(self):
        ds = Dataset.from_list([{"other": "x"}])
        with pytest.raises(ValueError, match="prompt"):
            validate_dataset_format(ds, "grpo")

    def test_invalid_prompt_raises_with_indices(self):
        ds = Dataset.from_list(
            [
                {"prompt": "ok"},
                {"prompt": ""},
            ]
        )
        with pytest.raises(ValueError, match="invalid `prompt`"):
            validate_dataset_format(ds, "grpo")

    def test_vlm_grpo_with_good_samples_passes(self, monkeypatch):
        # Patch image loader to skip actual disk checks
        from leap_finetune.data_loading import image_loader

        monkeypatch.setattr(image_loader, "is_image_loadable", lambda s: True)

        ds = Dataset.from_list(
            [
                {
                    "prompt": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": "/a.jpg"},
                                {"type": "text", "text": "q"},
                            ],
                        }
                    ]
                }
            ]
        )
        out = validate_dataset_format(ds, "vlm_grpo")
        assert out is ds

    def test_vlm_grpo_missing_prompt_column_raises(self):
        ds = Dataset.from_list([{"messages": [{"role": "user", "content": []}]}])
        with pytest.raises(ValueError, match="prompt"):
            validate_dataset_format(ds, "vlm_grpo")


# === SFT to GRPO normalization: messages split into prompt + solution ===


class TestVLMGRPOSFTNormalization:
    """When a VLM SFT dataset (messages with user + assistant turns) is
    used for vlm_grpo, the normalizer must split the messages into
    ``prompt`` (user/system turns only) and ``solution`` (last assistant
    turn text content).
    """

    def test_sft_messages_split_into_prompt_and_solution(self):
        n = normalize_columns("vlm_grpo")
        row = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "/img.jpg"},
                        {"type": "text", "text": "find the cat"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": '[{"label": "cat", "bbox": [0.1, 0.2, 0.5, 0.6]}]',
                        },
                    ],
                },
            ]
        }
        result = n(row)
        # prompt should contain only the user turn
        assert len(result["prompt"]) == 1
        assert result["prompt"][0]["role"] == "user"
        # solution should be the assistant text
        assert result["solution"] == '[{"label": "cat", "bbox": [0.1, 0.2, 0.5, 0.6]}]'
        # original messages column should be gone
        assert "messages" not in result

    def test_sft_string_content_assistant(self):
        """Assistant content as plain string (non-VLM SFT format)."""
        n = normalize_columns("vlm_grpo")
        row = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": "the answer"},
            ]
        }
        result = n(row)
        assert result["solution"] == "the answer"

    def test_existing_solution_not_overwritten(self):
        """If the row already has a solution column, don't overwrite it."""
        n = normalize_columns("vlm_grpo")
        row = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "q"}]},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "from assistant"}],
                },
            ],
            "solution": "explicit solution",
        }
        result = n(row)
        assert result["solution"] == "explicit solution"

    def test_system_message_preserved_in_prompt(self):
        n = normalize_columns("vlm_grpo")
        row = {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are helpful."}],
                },
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
            ]
        }
        result = n(row)
        assert len(result["prompt"]) == 2
        assert result["prompt"][0]["role"] == "system"
        assert result["prompt"][1]["role"] == "user"

    def test_ndarray_columns_converted_to_list(self):
        """Parquet deserialization returns numpy ndarrays for list columns."""
        import numpy as np

        n = normalize_columns("vlm_grpo")
        messages = np.array(
            [
                {"role": "user", "content": [{"type": "text", "text": "q"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
            ],
            dtype=object,
        )
        row = {"messages": messages}
        result = n(row)
        assert isinstance(result["prompt"], list)
        assert result["solution"] == "a"

    def test_user_only_messages_no_solution(self):
        """Messages without assistant turn: prompt set, no solution."""
        n = normalize_columns("vlm_grpo")
        row = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "q"}]},
            ]
        }
        result = n(row)
        assert len(result["prompt"]) == 1
        assert "solution" not in result

    def test_native_grpo_format_untouched(self):
        """Row with prompt column already set: no splitting occurs."""
        n = normalize_columns("vlm_grpo")
        row = {
            "prompt": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "q"}],
                }
            ],
            "solution": "existing",
        }
        result = n(row)
        assert result["prompt"][0]["role"] == "user"
        assert result["solution"] == "existing"

    def test_image_root_applied_after_split(self):
        n = normalize_columns("vlm_grpo", image_root="/data")
        row = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "rel/img.jpg"},
                        {"type": "text", "text": "describe"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "a dog"}],
                },
            ]
        }
        result = n(row)
        assert result["prompt"][0]["content"][0]["image"] == "/data/rel/img.jpg"
        assert result["solution"] == "a dog"


# === VLM Grounding reward functions ===

_SPEC_JSON = importlib.util.spec_from_file_location(
    "vlm_grounding_recipe",
    pathlib.Path(__file__).parents[2]
    / "rewards"
    / "tasks"
    / "vlm_grounding"
    / "recipe.py",
)
VGJ = importlib.util.module_from_spec(_SPEC_JSON)
_SPEC_JSON.loader.exec_module(VGJ)


def _assistant(content: str):
    """Short helper to build a conversational completion."""
    return [{"role": "assistant", "content": content}]


class TestVLMGroundingJSONExtract:
    """Test JSON-only validation in ``_extract_bboxes`` (no tags)."""

    # --- happy path ---

    def test_valid_bare_json_array(self):
        text = '[{"label": "cat", "bbox": [0.1, 0.2, 0.5, 0.6]}]'
        assert VGJ._extract_bboxes(text) == ([[0.1, 0.2, 0.5, 0.6]], 1)

    def test_trailing_whitespace_allowed(self):
        text = '[{"label": "a", "bbox": [0.0, 0.0, 1.0, 1.0]}]\n\n'
        assert VGJ._extract_bboxes(text) == ([[0.0, 0.0, 1.0, 1.0]], 1)

    def test_leading_whitespace_allowed(self):
        text = '\n  [{"label": "a", "bbox": [0.0, 0.0, 1.0, 1.0]}]'
        assert VGJ._extract_bboxes(text) == ([[0.0, 0.0, 1.0, 1.0]], 1)

    # --- collapse modes ---

    def test_reject_repetition_collapse(self):
        """The model emitting the same answer twice is invalid JSON."""
        text = (
            '[{"label": "a", "bbox": [0.1, 0.2, 0.5, 0.6]}]'
            '[{"label": "a", "bbox": [0.1, 0.2, 0.5, 0.6]}]'
        )
        assert VGJ._extract_bboxes(text) is None

    def test_reject_trailing_text(self):
        text = '[{"label": "a", "bbox": [0.1, 0.2, 0.5, 0.6]}] some reasoning'
        assert VGJ._extract_bboxes(text) is None

    def test_reject_leading_text(self):
        text = 'I see a cat. [{"label": "a", "bbox": [0.1, 0.2, 0.5, 0.6]}]'
        assert VGJ._extract_bboxes(text) is None

    def test_reject_legacy_answer_tags(self):
        """Tag-wrapped output is no longer accepted (the model shouldn't
        emit them and the data prep no longer asks for them).
        """
        text = '<answer>[{"label": "a", "bbox": [0.1, 0.2, 0.5, 0.6]}]</answer>'
        assert VGJ._extract_bboxes(text) is None

    def test_reject_malformed_json(self):
        text = '[{"label": "a", "bbox": [0.1, 0.2, 0.5, 0.6}]'
        assert VGJ._extract_bboxes(text) is None

    def test_reject_empty_string(self):
        assert VGJ._extract_bboxes("") is None

    # --- structure checks ---

    def test_reject_bare_dict_no_outer_array(self):
        text = '{"label": "a", "bbox": [0.1, 0.2, 0.5, 0.6]}'
        assert VGJ._extract_bboxes(text) is None

    def test_reject_empty_array(self):
        assert VGJ._extract_bboxes("[]") is None

    def test_multi_object_array_returns_all_with_count(self):
        text = (
            '[{"label": "a", "bbox": [0.1, 0.2, 0.5, 0.6]}, '
            '{"label": "b", "bbox": [0.2, 0.3, 0.6, 0.7]}]'
        )
        result = VGJ._extract_bboxes(text)
        assert result is not None
        bboxes, count = result
        assert len(bboxes) == 2
        assert count == 2

    def test_partial_malformed_counts_total(self):
        text = (
            '[{"label": "a", "bbox": [0.1, 0.2, 0.5, 0.6]}, '
            '{"label": "b"}, '
            '{"label": "c", "bbox": "not-a-list"}]'
        )
        result = VGJ._extract_bboxes(text)
        assert result is not None
        bboxes, count = result
        assert len(bboxes) == 1
        assert count == 3

    def test_reject_list_of_non_dict(self):
        assert VGJ._extract_bboxes("[[0.1, 0.2, 0.3, 0.4]]") is None

    # --- bbox field checks ---

    def test_reject_missing_bbox_field(self):
        assert VGJ._extract_bboxes('[{"label": "a"}]') is None

    def test_reject_wrong_bbox_length(self):
        assert VGJ._extract_bboxes('[{"bbox": [0.1, 0.2]}]') is None

    def test_reject_non_numeric_bbox(self):
        assert VGJ._extract_bboxes('[{"bbox": ["a", "b", "c", "d"]}]') is None

    def test_reject_inverted_bbox(self):
        assert VGJ._extract_bboxes('[{"bbox": [0.5, 0.5, 0.1, 0.1]}]') is None

    def test_reject_nan_bbox(self):
        assert VGJ._extract_bboxes('[{"bbox": [NaN, 0.1, 0.5, 0.5]}]') is None


# === Text GRPO task bundles ===

_TASKS_ROOT = pathlib.Path(__file__).parents[2] / "rewards" / "tasks"

_SPEC_GSM8K = importlib.util.spec_from_file_location(
    "gsm8k_recipe",
    _TASKS_ROOT / "gsm8k" / "recipe.py",
)
GSM8K_MOD = importlib.util.module_from_spec(_SPEC_GSM8K)
_SPEC_GSM8K.loader.exec_module(GSM8K_MOD)

_SPEC_MCQA = importlib.util.spec_from_file_location(
    "mcqa_recipe",
    _TASKS_ROOT / "mcqa" / "recipe.py",
)
MCQA_MOD = importlib.util.module_from_spec(_SPEC_MCQA)
_SPEC_MCQA.loader.exec_module(MCQA_MOD)

_SPEC_IFEVAL = importlib.util.spec_from_file_location(
    "ifeval_recipe",
    _TASKS_ROOT / "ifeval" / "recipe.py",
)
IFEVAL_MOD = importlib.util.module_from_spec(_SPEC_IFEVAL)
_SPEC_IFEVAL.loader.exec_module(IFEVAL_MOD)


class TestGSM8KReward:
    def test_strict_marker_match(self):
        c = _assistant("Step 1: 48/2 = 24. Step 2: 48 + 24 = 72.\n#### 72")
        assert GSM8K_MOD.gsm8k_reward([c], solution=["72"]) == [1.0]

    def test_fallback_last_number(self):
        c = _assistant("The answer, after careful thought, is 72.")
        assert GSM8K_MOD.gsm8k_reward([c], solution=["72"]) == [1.0]

    def test_incorrect(self):
        c = _assistant("#### 10")
        assert GSM8K_MOD.gsm8k_reward([c], solution=["72"]) == [0.0]

    def test_comma_normalization(self):
        c = _assistant("#### 1,000")
        assert GSM8K_MOD.gsm8k_reward([c], solution=["1000"]) == [1.0]

    def test_gt_is_full_cot(self):
        """When `solution` is the full assistant CoT (from auto-split),
        the reward still extracts the `#### N` marker from it."""
        c = _assistant("#### 72")
        gt = "Natalia sold 48/2 = 24 clips.\nShe sold 48+24 = 72 altogether.\n#### 72"
        assert GSM8K_MOD.gsm8k_reward([c], solution=[gt]) == [1.0]

    def test_missing_solution_skipped(self):
        c = _assistant("#### 42")
        assert GSM8K_MOD.gsm8k_reward([c], solution=None) == [None]

    def test_string_completion(self):
        """TRL can pass raw strings for non-conversational prompts."""
        assert GSM8K_MOD.gsm8k_reward(["#### 42"], solution=["42"]) == [1.0]


class TestMCQAReward:
    def test_answer_colon(self):
        c = _assistant("After analysis, Answer: B")
        assert MCQA_MOD.mcqa_reward([c], solution=["B"]) == [1.0]

    def test_markdown_bold(self):
        c = _assistant("**Answer:** C")
        assert MCQA_MOD.mcqa_reward([c], solution=["C"]) == [1.0]

    def test_boxed(self):
        c = _assistant("After reasoning the answer is \\boxed{D}")
        assert MCQA_MOD.mcqa_reward([c], solution=["D"]) == [1.0]

    def test_last_match_wins(self):
        """Self-correction: the final answer should score, not the first."""
        c = _assistant("At first I thought Answer: A, but Answer: B")
        assert MCQA_MOD.mcqa_reward([c], solution=["B"]) == [1.0]

    def test_incorrect(self):
        c = _assistant("Answer: A")
        assert MCQA_MOD.mcqa_reward([c], solution=["B"]) == [0.0]

    def test_gt_is_full_sentence(self):
        """`solution` coming from auto-split might be 'Answer: B'."""
        c = _assistant("Answer: B")
        assert MCQA_MOD.mcqa_reward([c], solution=["Answer: B"]) == [1.0]

    def test_missing_solution_skipped(self):
        c = _assistant("Answer: B")
        assert MCQA_MOD.mcqa_reward([c], solution=None) == [None]

    def test_invalid_gt_skipped(self):
        c = _assistant("Answer: B")
        assert MCQA_MOD.mcqa_reward([c], solution=[""])[0] is None


class TestIFEvalReward:
    def test_no_comma_pass(self):
        c = _assistant("A response without that punctuation mark.")
        sol = ['[{"instruction_id": ["punctuation:no_comma"], "kwargs": [null]}]']
        assert IFEVAL_MOD.ifeval_reward([c], solution=sol) == [1.0]

    def test_no_comma_fail(self):
        c = _assistant("Commas, like these, are forbidden here.")
        sol = ['[{"instruction_id": ["punctuation:no_comma"], "kwargs": [null]}]']
        assert IFEVAL_MOD.ifeval_reward([c], solution=sol) == [0.0]

    def test_num_words_at_least(self):
        c = _assistant("word " * 50)
        sol = [
            '[{"instruction_id": ["length_constraints:number_words"], '
            '"kwargs": [{"relation": "at least", "num_words": 20}]}]'
        ]
        assert IFEVAL_MOD.ifeval_reward([c], solution=sol) == [1.0]

    def test_num_words_at_least_fail(self):
        c = _assistant("only three words")
        sol = [
            '[{"instruction_id": ["length_constraints:number_words"], '
            '"kwargs": [{"relation": "at least", "num_words": 10}]}]'
        ]
        assert IFEVAL_MOD.ifeval_reward([c], solution=sol) == [0.0]

    def test_highlighted_sections(self):
        c = _assistant("Here *first highlight* and *second highlight* and *third*.")
        sol = [
            '[{"instruction_id": ["detectable_format:number_highlighted_sections"], '
            '"kwargs": [{"num_highlights": 3}]}]'
        ]
        assert IFEVAL_MOD.ifeval_reward([c], solution=sol) == [1.0]

    def test_keywords_multiple(self):
        c = _assistant("A kaleidoscope nebula whisper labyrinth paradox.")
        sol = [
            '[{"instruction_id": ["count:keywords_multiple"], '
            '"kwargs": [{"keyword1": "kaleidoscope", "keyword2": "nebula", '
            '"keyword3": "whisper", "keyword4": "labyrinth", "keyword5": "paradox"}]}]'
        ]
        assert IFEVAL_MOD.ifeval_reward([c], solution=sol) == [1.0]

    def test_keywords_multiple_missing_one(self):
        c = _assistant("A kaleidoscope nebula whisper labyrinth.")
        sol = [
            '[{"instruction_id": ["count:keywords_multiple"], '
            '"kwargs": [{"keyword1": "kaleidoscope", "keyword2": "paradox"}]}]'
        ]
        assert IFEVAL_MOD.ifeval_reward([c], solution=sol) == [0.0]

    def test_letter_count_in_word(self):
        c = _assistant("There are 3 Rs in the word.")
        sol = [
            '[{"instruction_id": ["counting:letter_count_in_word"], '
            '"kwargs": [{"letter": "r", "word": "strawberry"}]}]'
        ]
        assert IFEVAL_MOD.ifeval_reward([c], solution=sol) == [1.0]

    def test_combined_constraints_partial(self):
        """One constraint passes and one fails, so the score is 0.5."""
        c = _assistant("Lots of words " * 30)
        sol = [
            '[{"instruction_id": ["punctuation:no_comma", '
            '"length_constraints:number_words"], '
            '"kwargs": [null, {"relation": "at least", "num_words": 20}]}]'
        ]
        assert IFEVAL_MOD.ifeval_reward([c], solution=sol) == [1.0]

    def test_unsupported_only_skipped(self):
        c = _assistant("anything")
        sol = ['[{"instruction_id": ["fake:unsupported"], "kwargs": [null]}]']
        assert IFEVAL_MOD.ifeval_reward([c], solution=sol) == [None]

    def test_malformed_json_skipped(self):
        c = _assistant("anything")
        assert IFEVAL_MOD.ifeval_reward([c], solution=["not json"]) == [None]

    def test_missing_solution_skipped(self):
        c = _assistant("anything")
        assert IFEVAL_MOD.ifeval_reward([c], solution=None) == [None]


class TestStrictFormatReward:
    """Strict-format gate: only well-formed bare JSON bbox arrays score 1.0."""

    def test_strict_format_reward_valid(self):
        c = _assistant('[{"label": "a", "bbox": [0, 0, 1, 1]}]')
        assert VGJ.strict_format_reward([c]) == [1.0]

    def test_strict_format_reward_rejects_all_collapse_modes(self):
        completions = [
            _assistant("just plain text"),
            _assistant('{"label": "a", "bbox": [0, 0, 1, 1]}'),  # bare dict
            _assistant(  # repetition: invalid JSON
                '[{"label": "a", "bbox": [0, 0, 1, 1]}]'
                '[{"label": "a", "bbox": [0, 0, 1, 1]}]'
            ),
            _assistant('[{"label": "a", "bbox": [0, 0, 1, 1]}] more reasoning'),
            _assistant(
                '<answer>[{"label": "a", "bbox": [0, 0, 1, 1]}]</answer>'
            ),  # legacy tags
        ]
        rewards = VGJ.strict_format_reward(completions)
        assert rewards == [0.0, 0.0, 0.0, 0.0, 0.0]

    def test_gt_parsing_bare_json(self):
        gt = '[{"label": "a", "bbox": [0.1, 0.2, 0.5, 0.6]}]'
        assert VGJ._parse_gt_bboxes(gt) == [[0.1, 0.2, 0.5, 0.6]]

    def test_gt_parsing_rejects_garbage(self):
        assert VGJ._parse_gt_bboxes(None) == []
        assert VGJ._parse_gt_bboxes("") == []
        assert VGJ._parse_gt_bboxes("not json") == []
        assert VGJ._parse_gt_bboxes('{"not": "list"}') == []

    def test_iou_recipe_loads(self):
        from leap_finetune.rl.rewards import resolve_reward_specs

        funcs, _ = resolve_reward_specs(
            {
                "recipe": "./rewards/tasks/vlm_grounding/recipe.py::VLMGroundingIoURecipe"
            },
            config_dir=".",
        )
        assert [f.__name__ for f in funcs] == ["strict_format_reward", "iou_f1_reward"]


class TestVLMGroundingIoUF1:
    """Test the Hungarian-matched soft F1 reward."""

    @staticmethod
    def _c(text: str):
        return [{"role": "assistant", "content": text}]

    def test_single_pred_exact_match(self):
        c = self._c('[{"label": "cat", "bbox": [0.1, 0.1, 0.5, 0.5]}]')
        s = '[{"label": "cat", "bbox": [0.1, 0.1, 0.5, 0.5]}]'
        [r] = VGJ.iou_f1_reward([c], solution=[s])
        assert r == pytest.approx(1.0, abs=1e-6)

    def test_single_pred_partial_iou(self):
        # pred [0, 0, 0.5, 0.5] ∩ gt [0.25, 0, 0.75, 0.5]
        # intersection = 0.25 * 0.5 = 0.125; union = 0.25 + 0.25 - 0.125 = 0.375
        # IoU = 1/3; single pred single GT means F1 = IoU.
        c = self._c('[{"label": "cat", "bbox": [0.0, 0.0, 0.5, 0.5]}]')
        s = '[{"label": "cat", "bbox": [0.25, 0.0, 0.75, 0.5]}]'
        [r] = VGJ.iou_f1_reward([c], solution=[s])
        assert r == pytest.approx(1 / 3, rel=1e-3)

    def test_multiple_preds_single_gt_penalizes_extras(self):
        # 3 preds, one perfect hit, two wrong. Best match IoU = 1.0.
        # sum_iou = 1.0 (only one match possible); P = 1/3, R = 1; F1 = 0.5
        c = self._c(
            '[{"label": "cat", "bbox": [0.1, 0.1, 0.5, 0.5]},'
            ' {"label": "dog", "bbox": [0.6, 0.6, 0.8, 0.8]},'
            ' {"label": "bird", "bbox": [0.0, 0.0, 0.2, 0.2]}]'
        )
        s = '[{"label": "cat", "bbox": [0.1, 0.1, 0.5, 0.5]}]'
        [r] = VGJ.iou_f1_reward([c], solution=[s])
        assert r == pytest.approx(0.5, abs=1e-6)

    def test_multiple_gts_all_perfect(self):
        c = self._c(
            '[{"label": "cat", "bbox": [0.1, 0.1, 0.3, 0.3]},'
            ' {"label": "dog", "bbox": [0.5, 0.5, 0.7, 0.7]}]'
        )
        s = (
            '[{"label": "cat", "bbox": [0.1, 0.1, 0.3, 0.3]},'
            ' {"label": "dog", "bbox": [0.5, 0.5, 0.7, 0.7]}]'
        )
        [r] = VGJ.iou_f1_reward([c], solution=[s])
        assert r == pytest.approx(1.0, abs=1e-6)

    def test_hungarian_resolves_swapped_order(self):
        # Pred in reversed order; Hungarian should still match both perfectly.
        c = self._c(
            '[{"label": "dog", "bbox": [0.5, 0.5, 0.7, 0.7]},'
            ' {"label": "cat", "bbox": [0.1, 0.1, 0.3, 0.3]}]'
        )
        s = (
            '[{"label": "cat", "bbox": [0.1, 0.1, 0.3, 0.3]},'
            ' {"label": "dog", "bbox": [0.5, 0.5, 0.7, 0.7]}]'
        )
        [r] = VGJ.iou_f1_reward([c], solution=[s])
        assert r == pytest.approx(1.0, abs=1e-6)

    def test_parse_fail_returns_zero(self):
        c = self._c("not valid json")
        s = '[{"label": "cat", "bbox": [0.1, 0.1, 0.5, 0.5]}]'
        [r] = VGJ.iou_f1_reward([c], solution=[s])
        assert r == 0.0

    def test_degenerate_box_silently_dropped(self):
        # One valid pred, one degenerate (x2 < x1). Reward uses only valid.
        c = self._c(
            '[{"label": "cat", "bbox": [0.1, 0.1, 0.5, 0.5]},'
            ' {"label": "bad", "bbox": [0.8, 0.8, 0.3, 0.3]}]'
        )
        s = '[{"label": "cat", "bbox": [0.1, 0.1, 0.5, 0.5]}]'
        [r] = VGJ.iou_f1_reward([c], solution=[s])
        assert r == pytest.approx(1.0, abs=1e-6)

    def test_hallucination_no_gt(self):
        c = self._c('[{"label": "cat", "bbox": [0.1, 0.1, 0.5, 0.5]}]')
        s = "[]"
        [r] = VGJ.iou_f1_reward([c], solution=[s])
        assert r == 0.0

    def test_correct_abstention(self):
        # Empty list pred means _extract_bboxes returns None, so num_pred = 0.
        # Empty list GT means num_gt = 0. Both empty means abstention correct.
        c = self._c("[]")
        s = "[]"
        [r] = VGJ.iou_f1_reward([c], solution=[s])
        assert r == pytest.approx(1.0)

    def test_solution_none_returns_zeros(self):
        c = self._c('[{"label": "cat", "bbox": [0.1, 0.1, 0.5, 0.5]}]')
        rewards = VGJ.iou_f1_reward([c, c], solution=None)
        assert rewards == [0.0, 0.0]

    def test_batch_independent(self):
        c_good = self._c('[{"label": "cat", "bbox": [0.1, 0.1, 0.5, 0.5]}]')
        c_bad = self._c("broken")
        s_good = '[{"label": "cat", "bbox": [0.1, 0.1, 0.5, 0.5]}]'
        s_other = '[{"label": "dog", "bbox": [0.0, 0.0, 1.0, 1.0]}]'
        r_good, r_bad = VGJ.iou_f1_reward([c_good, c_bad], solution=[s_good, s_other])
        assert r_good == pytest.approx(1.0)
        assert r_bad == 0.0

    def test_hungarian_direct_1to1(self):
        preds = [[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]]
        gts = [[0.5, 0.5, 0.7, 0.7], [0.1, 0.1, 0.3, 0.3]]
        matches = VGJ._hungarian_match(preds, gts)
        assert len(matches) == 2
        assert all(iou == pytest.approx(1.0) for _, _, iou in matches)
        used_p = {p for p, _, _ in matches}
        used_g = {g for _, g, _ in matches}
        assert used_p == {0, 1}
        assert used_g == {0, 1}

    def test_hungarian_empty_inputs(self):
        assert VGJ._hungarian_match([], [[0, 0, 1, 1]]) == []
        assert VGJ._hungarian_match([[0, 0, 1, 1]], []) == []

    def test_recipe_uses_iou_f1(self):
        from leap_finetune.rl.rewards import resolve_reward_specs

        funcs, weights = resolve_reward_specs(
            {
                "recipe": "./rewards/tasks/vlm_grounding/recipe.py::VLMGroundingIoURecipe"
            },
            config_dir=".",
        )
        names = {f.__name__ for f in funcs}
        assert "iou_f1_reward" in names
        assert "strict_format_reward" in names
        assert weights == [0.1, 1.0]


class TestVLMGroundingCIoUF1:
    """CIoU-F1 mirror of the IoU-F1 tests.

    The F1 structure (Hungarian 1-to-1 + precision/recall aggregator)
    is identical to :func:`iou_f1_reward`; only the per-pair geometry
    changes from IoU to CIoU. For axis-aligned, perfectly-overlapping
    pairs CIoU = IoU = 1, so the happy-path rewards should match 1.0
    identically. For partial-overlap pairs CIoU < IoU because of the
    center-distance penalty; we don't assert exact values there, just
    that the reward stays in [0, 1] and behaves monotonically with
    alignment quality.
    """

    @staticmethod
    def _c(text: str):
        return [{"role": "assistant", "content": text}]

    def test_single_pred_exact_match(self):
        # Identical boxes mean IoU = 1, center_dist = 0, and CIoU = 1.
        c = self._c('[{"label": "cat", "bbox": [0.1, 0.1, 0.5, 0.5]}]')
        s = '[{"label": "cat", "bbox": [0.1, 0.1, 0.5, 0.5]}]'
        [r] = VGJ.ciou_f1_reward([c], solution=[s])
        assert r == pytest.approx(1.0, abs=1e-6)

    def test_ciou_is_stricter_than_iou_for_offset_boxes(self):
        """For a partial-overlap pair with offset centers, CIoU < IoU.

        The IoU-F1 reward for this case is exactly 1/3. CIoU-F1 should
        be strictly less because of the center-distance penalty.
        """
        c = self._c('[{"label": "cat", "bbox": [0.0, 0.0, 0.5, 0.5]}]')
        s = '[{"label": "cat", "bbox": [0.25, 0.0, 0.75, 0.5]}]'
        [r_iou] = VGJ.iou_f1_reward([c], solution=[s])
        [r_ciou] = VGJ.ciou_f1_reward([c], solution=[s])
        assert r_iou == pytest.approx(1 / 3, rel=1e-3)
        assert 0.0 <= r_ciou < r_iou, (
            "CIoU-F1 must penalize center-offset more aggressively than IoU-F1"
        )

    def test_multiple_gts_all_perfect(self):
        # Two perfect matches on two GTs means CIoU = 1 for both and F1 = 1.
        c = self._c(
            '[{"label": "cat", "bbox": [0.1, 0.1, 0.3, 0.3]},'
            ' {"label": "dog", "bbox": [0.5, 0.5, 0.7, 0.7]}]'
        )
        s = (
            '[{"label": "cat", "bbox": [0.1, 0.1, 0.3, 0.3]},'
            ' {"label": "dog", "bbox": [0.5, 0.5, 0.7, 0.7]}]'
        )
        [r] = VGJ.ciou_f1_reward([c], solution=[s])
        assert r == pytest.approx(1.0, abs=1e-6)

    def test_hungarian_resolves_swapped_order(self):
        # Reverse order still matches both through Hungarian assignment.
        c = self._c(
            '[{"label": "dog", "bbox": [0.5, 0.5, 0.7, 0.7]},'
            ' {"label": "cat", "bbox": [0.1, 0.1, 0.3, 0.3]}]'
        )
        s = (
            '[{"label": "cat", "bbox": [0.1, 0.1, 0.3, 0.3]},'
            ' {"label": "dog", "bbox": [0.5, 0.5, 0.7, 0.7]}]'
        )
        [r] = VGJ.ciou_f1_reward([c], solution=[s])
        assert r == pytest.approx(1.0, abs=1e-6)

    def test_parse_fail_returns_zero(self):
        c = self._c("not valid json")
        s = '[{"label": "cat", "bbox": [0.1, 0.1, 0.5, 0.5]}]'
        [r] = VGJ.ciou_f1_reward([c], solution=[s])
        assert r == 0.0

    def test_hallucination_no_gt(self):
        c = self._c('[{"label": "cat", "bbox": [0.1, 0.1, 0.5, 0.5]}]')
        s = "[]"
        [r] = VGJ.ciou_f1_reward([c], solution=[s])
        assert r == 0.0

    def test_correct_abstention(self):
        c = self._c("[]")
        s = "[]"
        [r] = VGJ.ciou_f1_reward([c], solution=[s])
        assert r == pytest.approx(1.0)

    def test_solution_none_returns_zeros(self):
        c = self._c('[{"label": "cat", "bbox": [0.1, 0.1, 0.5, 0.5]}]')
        rewards = VGJ.ciou_f1_reward([c, c], solution=None)
        assert rewards == [0.0, 0.0]

    def test_disjoint_pair_clamped_not_negative(self):
        """Disjoint boxes with different aspect ratio produce raw CIoU < 0.

        The aggregator must clamp to 0 so precision/recall stay in
        [0, 1] and the final F1 is 0, not a negative number.
        """
        c = self._c('[{"label": "cat", "bbox": [0.0, 0.0, 0.1, 0.4]}]')
        # Disjoint, very different aspect
        s = '[{"label": "cat", "bbox": [0.8, 0.9, 1.0, 0.95]}]'
        [r] = VGJ.ciou_f1_reward([c], solution=[s])
        assert r == 0.0

    def test_hungarian_direct_ciou(self):
        """The parametrized matcher with ``similarity=_ciou`` returns
        CIoU values on perfect matches, not IoU."""
        preds = [[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]]
        gts = [[0.5, 0.5, 0.7, 0.7], [0.1, 0.1, 0.3, 0.3]]
        matches = VGJ._hungarian_match(preds, gts, similarity=VGJ._ciou)
        assert len(matches) == 2
        assert all(ciou == pytest.approx(1.0) for _, _, ciou in matches)
        used_p = {p for p, _, _ in matches}
        used_g = {g for _, g, _ in matches}
        assert used_p == {0, 1}
        assert used_g == {0, 1}

    def test_iou_matcher_backcompat_unchanged(self):
        """The default ``similarity=_iou`` path must still return raw
        IoUs (not clamped or transformed) so :func:`iou_f1_reward`'s
        numerical output is bit-exact preserved by the refactor."""
        preds = [[0.0, 0.0, 0.5, 0.5]]
        gts = [[0.25, 0.0, 0.75, 0.5]]
        matches = VGJ._hungarian_match(preds, gts)
        assert len(matches) == 1
        assert matches[0][2] == pytest.approx(1 / 3, rel=1e-3)

    def test_recipe_uses_ciou_f1(self):
        from leap_finetune.rl.rewards import resolve_reward_specs

        funcs, weights = resolve_reward_specs(
            {
                "recipe": "./rewards/tasks/vlm_grounding/recipe.py::VLMGroundingCIoURecipe"
            },
            config_dir=".",
        )
        names = {f.__name__ for f in funcs}
        assert "ciou_f1_reward" in names
        assert "strict_format_reward" in names
        assert "iou_f1_reward" not in names
        assert weights == [0.1, 1.0]


# === VLM GRPO trainer override: image lift-to-top-level ===
#
# Regression test for the critical fix in LFMVLMGRPOTrainer._generate_and_score_completions:
# TRL's GRPOTrainer looks for a top-level `images` column on each example
# to build the training forward-pass pixel_values. Our VLM GRPO dataset
# embeds images inside the `prompt` messages, so without the override TRL
# detects no images, runs the training forward without pixel_values, and
# the LFM2-VL model silently falls back to treating `<image>` placeholder
# tokens as raw text embeddings. The rollout/training log-prob gap blows
# up (~0.3/token) and the policy degrades.


class TestVLMGRPOImageLift:
    """Assert the trainer's image-lift wrapper surfaces images to TRL.

    These tests monkey-patch the parent ``_generate_and_score_completions``
    so we can isolate the wrapper logic without touching a real model.
    """

    def _build_instance(self):
        """Construct a bare LFMVLMGRPOTrainer without running __init__.

        GRPOTrainer.__init__ does heavy work (model loading, processor,
        accelerator). We only need to exercise the overridden method, so
        we bypass init and rely on ``super().`` dispatching via the MRO.
        """
        from leap_finetune.training.vlm_grpo import LFMVLMGRPOTrainer

        return LFMVLMGRPOTrainer.__new__(LFMVLMGRPOTrainer)

    def _make_row(self, image_field):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_field},
                        {"type": "text", "text": "where is the cat?"},
                    ],
                }
            ],
            "solution": '[{"label": "cat", "bbox": [0.1, 0.2, 0.5, 0.6]}]',
        }

    def test_images_lifted_from_prompt_content(self, monkeypatch, tmp_path):
        from PIL import Image

        from trl import GRPOTrainer

        img_path = tmp_path / "cat.jpg"
        Image.new("RGB", (16, 16), color="red").save(img_path)

        captured = {}

        def fake_parent(self, inputs):
            captured["inputs"] = inputs
            return {"ok": True}

        monkeypatch.setattr(
            GRPOTrainer,
            "_generate_and_score_completions",
            fake_parent,
        )

        instance = self._build_instance()
        inputs = [self._make_row(str(img_path))]
        instance._generate_and_score_completions(inputs)

        assert "images" in captured["inputs"][0], (
            "TRL needs a top-level `images` column to build pixel_values "
            "for the training forward pass"
        )
        lifted = captured["inputs"][0]["images"]
        assert len(lifted) == 1
        assert isinstance(lifted[0], Image.Image), (
            "Image paths must be pre-loaded to PIL so `_tokenize_prompts` "
            "and the processor don't each re-open the file"
        )

    def test_existing_images_column_not_overwritten(self, monkeypatch, tmp_path):
        from PIL import Image

        from trl import GRPOTrainer

        existing = [Image.new("RGB", (8, 8), color="blue")]
        captured = {}

        def fake_parent(self, inputs):
            captured["inputs"] = inputs
            return None

        monkeypatch.setattr(
            GRPOTrainer,
            "_generate_and_score_completions",
            fake_parent,
        )

        instance = self._build_instance()
        row = self._make_row(str(tmp_path / "irrelevant.jpg"))
        row["images"] = existing
        instance._generate_and_score_completions([row])

        assert captured["inputs"][0]["images"] is existing

    def test_text_only_row_unchanged(self, monkeypatch):
        from trl import GRPOTrainer

        captured = {}

        def fake_parent(self, inputs):
            captured["inputs"] = inputs
            return None

        monkeypatch.setattr(
            GRPOTrainer,
            "_generate_and_score_completions",
            fake_parent,
        )

        instance = self._build_instance()
        text_only = {
            "prompt": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "solution": "hello",
        }
        instance._generate_and_score_completions([text_only])

        assert "images" not in captured["inputs"][0]

    def test_multi_image_order_preserved(self, monkeypatch, tmp_path):
        from PIL import Image

        from trl import GRPOTrainer

        # PNG (lossless) so the test can distinguish the two images by
        # pixel value without JPEG compression drift.
        p1 = tmp_path / "a.png"
        p2 = tmp_path / "b.png"
        Image.new("RGB", (8, 8), color="red").save(p1)
        Image.new("RGB", (8, 8), color="blue").save(p2)

        captured = {}

        def fake_parent(self, inputs):
            captured["inputs"] = inputs
            return None

        monkeypatch.setattr(
            GRPOTrainer,
            "_generate_and_score_completions",
            fake_parent,
        )

        instance = self._build_instance()
        row = {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(p1)},
                        {"type": "image", "image": str(p2)},
                        {"type": "text", "text": "compare"},
                    ],
                }
            ],
            "solution": "[]",
        }
        instance._generate_and_score_completions([row])

        lifted = captured["inputs"][0]["images"]
        assert len(lifted) == 2
        # Order must match content order so TRL's prepare_multimodal_messages
        # assigns the right image to the right placeholder.
        assert lifted[0].getpixel((0, 0)) == (255, 0, 0)
        assert lifted[1].getpixel((0, 0)) == (0, 0, 255)


class TestVLMGRPOSpatialShapesAlias:
    """Regression test for the spatial_shapes to image_sizes alias.

    LFM2-VL's processor returns ``spatial_shapes``, which TRL's GRPO
    trainer does NOT recognize in its multimodal whitelist. We
    piggyback on TRL's whitelisted ``image_sizes`` field instead:
    rename at the processor output so TRL propagates it through
    _generate_and_score_completions -> _compute_loss ->
    _get_per_token_logps_and_entropies, then rename back before
    calling the model forward (inside our
    _get_per_token_logps_and_entropies override).

    This test exercises the processor-wrapping context manager with
    a stub processor; it does NOT load a real LFM2-VL model.
    """

    def _make_stub_processor(self, output: dict):
        """Return a callable object whose ``__call__`` returns ``output``."""

        class _StubProcessor:
            def __call__(self, *args, **kwargs):
                # Return a fresh copy so mutation by the alias doesn't
                # bleed between test invocations.
                return dict(output)

        return _StubProcessor()

    def _build_instance(self, processor):
        from leap_finetune.training.vlm_grpo import LFMVLMGRPOTrainer

        instance = LFMVLMGRPOTrainer.__new__(LFMVLMGRPOTrainer)
        instance.processing_class = processor
        return instance

    def test_context_manager_renames_spatial_shapes(self):
        stub_output = {
            "input_ids": "fake",
            "pixel_values": "fake",
            "spatial_shapes": "fake_shapes",
        }
        processor = self._make_stub_processor(stub_output)
        instance = self._build_instance(processor)

        # Before the context manager: raw call returns spatial_shapes.
        assert "spatial_shapes" in processor()
        assert "image_sizes" not in processor()

        # Inside the context: the processor renames on output.
        with instance._aliasing_spatial_shapes_as_image_sizes():
            result = processor()
        assert "image_sizes" in result
        assert result["image_sizes"] == "fake_shapes"
        assert "spatial_shapes" not in result

        # After exit: original class behavior restored.
        assert "spatial_shapes" in processor()
        assert "image_sizes" not in processor()

    def test_context_manager_noop_when_no_processor(self):
        """Unit-test construction via ``__new__`` has no processor; the
        context manager must be a safe no-op rather than AttributeError."""
        from leap_finetune.training.vlm_grpo import LFMVLMGRPOTrainer

        instance = LFMVLMGRPOTrainer.__new__(LFMVLMGRPOTrainer)
        # No processing_class attribute set
        with instance._aliasing_spatial_shapes_as_image_sizes():
            pass  # just verify it doesn't raise

    def test_rename_back_in_get_per_token_logps(self, monkeypatch):
        """The override of ``_get_per_token_logps_and_entropies`` must
        undo the alias at the model-forward boundary: it receives
        ``image_sizes`` (from TRL's propagated dict) and must forward
        it to the model as ``spatial_shapes``."""
        from leap_finetune.training.vlm_grpo import LFMVLMGRPOTrainer

        instance = LFMVLMGRPOTrainer.__new__(LFMVLMGRPOTrainer)
        # LFM2-VL's forward accepts these names:
        instance.model_kwarg_keys = {
            "input_ids",
            "attention_mask",
            "pixel_values",
            "spatial_shapes",
            "pixel_attention_mask",
            "use_cache",
        }
        instance.temperature = 1.0

        captured_model_inputs: dict = {}

        class _FakeModel:
            def __call__(self, **kwargs):
                captured_model_inputs.update(kwargs)
                # Fake logits shaped to let selective_log_softmax run
                return type("O", (), {"logits": torch.randn(1, 6, 10)})()

        # Trivial 1-sample batch with a 5-token completion
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        instance._get_per_token_logps_and_entropies(
            _FakeModel(),
            input_ids,
            attention_mask,
            logits_to_keep=5,
            # The fields TRL's _compute_loss explicitly passes; image_sizes
            # is the aliased spatial_shapes.
            pixel_values=torch.zeros(1, 3, 4, 4),
            image_sizes=torch.tensor([[2, 2]]),
            pixel_attention_mask=torch.ones(1, 4, 4),
        )

        assert "spatial_shapes" in captured_model_inputs, (
            "The override must rename image_sizes to spatial_shapes at "
            "the model-forward boundary so LFM2-VL's forward can use it"
        )
        assert "image_sizes" not in captured_model_inputs, (
            "image_sizes must be consumed, not passed through; LFM2-VL "
            "does not accept it"
        )
        assert torch.equal(
            captured_model_inputs["spatial_shapes"], torch.tensor([[2, 2]])
        )
        assert "pixel_values" in captured_model_inputs
        assert "pixel_attention_mask" in captured_model_inputs
