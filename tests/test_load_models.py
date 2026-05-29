import builtins
import sys
import types

import pytest

from leap_finetune.utils import load_models

pytestmark = pytest.mark.configs


# === Attention implementation selection ===


class TestAttentionImplementation:
    def test_sdpa_when_flash_attn_metadata_missing(self, monkeypatch):
        monkeypatch.setattr(load_models, "is_flash_attn_2_available", lambda: False)

        assert load_models._get_attn_implementation() == "sdpa"

    def test_sdpa_when_flash_attn_import_fails(self, monkeypatch):
        monkeypatch.setattr(load_models, "is_flash_attn_2_available", lambda: True)
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "flash_attn":
                raise ImportError("broken extension")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        assert load_models._get_attn_implementation() == "sdpa"

    def test_flash_attention_when_import_succeeds(self, monkeypatch):
        monkeypatch.setattr(load_models, "is_flash_attn_2_available", lambda: True)
        flash_attn = types.ModuleType("flash_attn")
        flash_attn.flash_attn_func = object()
        flash_attn.flash_attn_varlen_func = object()
        monkeypatch.setitem(sys.modules, "flash_attn", flash_attn)

        assert load_models._get_attn_implementation() == "flash_attention_2"
