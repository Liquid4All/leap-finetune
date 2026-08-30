import pytest
from datasets import Dataset


class TestKTOFormat:
    def test_missing_columns_rejected(self):
        from leap_finetune.data_loading.validate_dataset_format import (
            validate_kto_format,
        )

        ds = Dataset.from_list([{"prompt": "Hi", "completion": "Hello"}])
        with pytest.raises(ValueError, match="label"):
            validate_kto_format(ds)

    def test_empty_completion_rejected(self):
        from leap_finetune.data_loading.validate_dataset_format import (
            validate_kto_format,
        )

        ds = Dataset.from_list([{"prompt": "Hi", "completion": "", "label": True}])
        with pytest.raises(ValueError, match="empty prompt/completion"):
            validate_kto_format(ds)

    def test_valid_rows_pass(self):
        from leap_finetune.data_loading.validate_dataset_format import (
            validate_kto_format,
        )

        ds = Dataset.from_list(
            [
                {
                    "prompt": [{"role": "user", "content": "Capital of France?"}],
                    "completion": [{"role": "assistant", "content": "Paris."}],
                    "label": True,
                },
                {
                    "prompt": [{"role": "user", "content": "Capital of France?"}],
                    "completion": [{"role": "assistant", "content": "London."}],
                    "label": False,
                },
            ]
        )
        validate_kto_format(ds)

    def test_row_filter_accepts_valid_and_rejects_invalid(self):
        from leap_finetune.data_loading.validate_dataset_format import get_row_filter

        f = get_row_filter("kto")
        valid = {
            "prompt": [{"role": "user", "content": "Hi"}],
            "completion": [{"role": "assistant", "content": "Hello"}],
            "label": True,
        }
        assert f(valid) is True
        assert f({**valid, "label": None}) is False
        assert f({**valid, "label": "yes"}) is False
        assert f({**valid, "completion": []}) is False
        assert f({"prompt": "Hi", "completion": "Hello", "label": False}) is True

    def test_row_filter_rejects_foreign_tool_markers(self):
        from leap_finetune.data_loading.validate_dataset_format import get_row_filter

        f = get_row_filter("kto")
        assert (
            f(
                {
                    "prompt": "Get weather",
                    "completion": "<tool_call>x</tool_call>",
                    "label": True,
                }
            )
            is False
        )
