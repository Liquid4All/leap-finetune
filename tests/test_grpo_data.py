"""Schema validation and normalization tests for grpo / vlm_grpo datasets.

Run with: `uv run pytest --data tests/test_grpo_data.py -v`
"""

import pytest

from leap_finetune.data_loaders.validate_loader import (
    get_row_filter,
    normalize_columns,
    validate_dataset_format,
)

# Import Dataset at module level — it's already available via the existing
# data_loaders import chain (conftest.py imports torch too).
from datasets import Dataset

pytestmark = pytest.mark.data


# === Row filter: grpo (text) ===


class TestGRPORowFilter:
    def setup_method(self):
        self.f = get_row_filter("grpo")

    def test_string_prompt_accepted(self):
        assert self.f({"prompt": "What is 2+2?"}) is True

    def test_messages_prompt_accepted(self):
        assert (
            self.f({"prompt": [{"role": "user", "content": "hi"}]}) is True
        )

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
        from leap_finetune.data_loaders import image_loader

        self._orig = image_loader.is_image_loadable
        image_loader.is_image_loadable = lambda s: True

        # The filter closes over the import at call time; reset its resolved fn
        self.f = get_row_filter("vlm_grpo")

    def teardown_method(self):
        from leap_finetune.data_loaders import image_loader

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
    def test_valid_grpo_passes(self):
        ds = Dataset.from_list(
            [
                {"prompt": "q1", "ground_truth": "a1"},
                {"prompt": [{"role": "user", "content": "q2"}], "solution": "a2"},
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
                {"prompt": []},
            ]
        )
        with pytest.raises(ValueError, match="invalid `prompt`"):
            validate_dataset_format(ds, "grpo")

    def test_vlm_grpo_with_good_samples_passes(self, monkeypatch):
        # Patch image loader to skip actual disk checks
        from leap_finetune.data_loaders import image_loader

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
