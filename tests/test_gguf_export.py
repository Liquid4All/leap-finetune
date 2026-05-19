import pathlib
from unittest.mock import patch

import pytest

from leap_finetune.quantization.gguf_export import (
    ADAPTER_QUANTS,
    ALL_QUANTS,
    BUNDLED_CONVERT_HF,
    BUNDLED_CONVERT_LORA,
    DIRECT_QUANTS,
    QUANTIZE_QUANTS,
    is_adapter_path,
    resolve_quantize_binary,
    validate_model_path,
)


# === Quant type classification ===


def test_direct_and_quantize_quants_are_disjoint():
    assert DIRECT_QUANTS & QUANTIZE_QUANTS == set()


def test_all_quants_is_union():
    assert ALL_QUANTS == DIRECT_QUANTS | QUANTIZE_QUANTS


def test_adapter_quants_subset_of_direct():
    assert ADAPTER_QUANTS == DIRECT_QUANTS


# === Bundled scripts exist ===


def test_bundled_convert_hf_exists():
    assert BUNDLED_CONVERT_HF.exists()


def test_bundled_convert_lora_exists():
    assert BUNDLED_CONVERT_LORA.exists()


# === is_adapter_path ===


def test_is_adapter_path_true(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    assert is_adapter_path(tmp_path) is True


def test_is_adapter_path_false(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    assert is_adapter_path(tmp_path) is False


# === validate_model_path ===


def test_validate_model_path_not_found():
    with pytest.raises(FileNotFoundError, match="Path not found"):
        validate_model_path(pathlib.Path("/nonexistent/path"))


def test_validate_model_path_not_dir(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("hello")
    with pytest.raises(ValueError, match="Expected a directory"):
        validate_model_path(f)


def test_validate_model_path_no_config(tmp_path):
    with pytest.raises(ValueError, match="No config.json"):
        validate_model_path(tmp_path)


def test_validate_model_path_with_config(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    validate_model_path(tmp_path)


def test_validate_model_path_with_adapter_config(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    validate_model_path(tmp_path)


# === resolve_quantize_binary ===


def test_resolve_quantize_binary_no_source():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(
            FileNotFoundError, match="llama-quantize binary is required"
        ):
            resolve_quantize_binary(None)


def test_resolve_quantize_binary_nonexistent():
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_quantize_binary("/nonexistent/llama.cpp")


def test_resolve_quantize_binary_not_built(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        resolve_quantize_binary(str(tmp_path))


def test_resolve_quantize_binary_found(tmp_path):
    bin_dir = tmp_path / "build" / "bin"
    bin_dir.mkdir(parents=True)
    binary = bin_dir / "llama-quantize"
    binary.write_text("")
    result = resolve_quantize_binary(str(tmp_path))
    assert result == binary


def test_resolve_quantize_binary_from_env(tmp_path):
    bin_dir = tmp_path / "build" / "bin"
    bin_dir.mkdir(parents=True)
    binary = bin_dir / "llama-quantize"
    binary.write_text("")
    with patch.dict("os.environ", {"LLAMA_CPP_DIR": str(tmp_path)}):
        result = resolve_quantize_binary(None)
        assert result == binary
