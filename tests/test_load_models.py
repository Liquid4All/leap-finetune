import builtins
import sys
import types

import pytest

from leap_finetune.checkpointing import model_loading
from leap_finetune.checkpointing.model_loading import (
    _is_moe_model,
    _maybe_enable_grouped_mm,
    _resolve_chat_template,
    _resolve_model_id,
)

pytestmark = pytest.mark.configs


class DummyConfig:
    def __init__(self, model_type: str, architectures: list[str]) -> None:
        self.model_type = model_type
        self.architectures = architectures


class DummyModel:
    def __init__(self, model_type: str, architectures: list[str]) -> None:
        self.config = DummyConfig(model_type, architectures)
        self.called = False
        self.impl = None

    def set_experts_implementation(self, impl: str) -> None:
        self.called = True
        self.impl = impl


def test_sdpa_when_flash_attn_metadata_missing(monkeypatch):
    monkeypatch.setattr(model_loading, "is_flash_attn_2_available", lambda: False)

    assert model_loading._get_attn_implementation() == "sdpa"


def test_sdpa_when_flash_attn_import_fails(monkeypatch):
    monkeypatch.setattr(model_loading, "is_flash_attn_2_available", lambda: True)
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "flash_attn":
            raise ImportError("broken extension")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert model_loading._get_attn_implementation() == "sdpa"


def test_flash_attention_when_import_succeeds(monkeypatch):
    monkeypatch.setattr(model_loading, "is_flash_attn_2_available", lambda: True)
    flash_attn = types.ModuleType("flash_attn")
    flash_attn.flash_attn_func = object()
    flash_attn.flash_attn_varlen_func = object()
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn)

    assert model_loading._get_attn_implementation() == "flash_attention_2"


def test_resolve_model_id_expands_liquidai_short_name():
    assert _resolve_model_id("LFM2-24B-A2B") == "LiquidAI/LFM2-24B-A2B"


def test_resolve_model_id_keeps_qualified_hf_id():
    assert _resolve_model_id("LiquidAI/LFM2-24B-A2B") == "LiquidAI/LFM2-24B-A2B"


def test_resolve_model_id_keeps_other_qualified_hf_id():
    assert _resolve_model_id("some-org/some-model") == "some-org/some-model"


def test_resolve_model_id_keeps_remote_uri():
    assert _resolve_model_id("s3://bucket/model") == "s3://bucket/model"


def test_resolve_model_id_keeps_existing_local_dir(tmp_path):
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()

    assert _resolve_model_id(str(model_dir)) == str(model_dir)


def test_resolve_chat_template_explicit_override_wins():
    assert _resolve_chat_template(chat_template="custom-template") == "custom-template"


def test_resolve_chat_template_defaults_lfm25_nonlocal_model():
    template = _resolve_chat_template("LiquidAI/LFM2.5-1.2B-Instruct")

    assert template is not None
    assert "{%- generation -%}" in template
    assert "List of tools" in template


def test_resolve_chat_template_defaults_lfm2_24b_nonlocal_model():
    template = _resolve_chat_template("LiquidAI/LFM2-24B-A2B")

    assert template is not None
    assert "{%- generation -%}" in template


def test_resolve_chat_template_does_not_default_local_model(tmp_path):
    model_dir = tmp_path / "LiquidAI" / "LFM2.5-local"
    model_dir.mkdir(parents=True)

    assert _resolve_chat_template(str(model_dir)) is None


def test_grouped_mm_override_only_applies_to_moe_models():
    dense = DummyModel("lfm2", ["Lfm2ForCausalLM"])
    moe = DummyModel("lfm2_moe", ["Lfm2MoeForCausalLM"])

    assert not _is_moe_model(dense)
    assert _is_moe_model(moe)

    _maybe_enable_grouped_mm(dense)
    _maybe_enable_grouped_mm(moe)

    assert dense.called is False
    assert moe.called is True
    assert moe.impl == "grouped_mm"
