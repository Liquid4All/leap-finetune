from __future__ import annotations

import json
from pathlib import Path

import torch

from leap_finetune.utils.model_utils import (
    HF_EXPORT_MAX_SHARD_SIZE,
    _canonicalize_hf_export_state_dict,
    _save_hf_pretrained_model,
    _save_root_hf_export,
)


class ConfigStub:
    auto_map = {
        "AutoConfig": "configuration_lfm2_moe.Lfm2MoeConfig",
        "AutoModelForCausalLM": "modeling_lfm2_moe.Lfm2MoeForCausalLM",
    }

    def __init__(self, source_dir: Path):
        self._name_or_path = str(source_dir)

    def save_pretrained(self, output_dir: str) -> None:
        Path(output_dir, "config.json").write_text(
            json.dumps(
                {
                    "model_type": "lfm2_moe",
                    "auto_map": self.auto_map,
                    "max_position_embeddings": 131072,
                }
            )
        )


class GenerationConfigStub:
    def save_pretrained(self, output_dir: str) -> None:
        Path(output_dir, "generation_config.json").write_text(
            json.dumps({"eos_token_id": 7})
        )


class ModelStub:
    def __init__(self, source_dir: Path):
        self.config = ConfigStub(source_dir)
        self.generation_config = GenerationConfigStub()
        self.saved_state_dict = None
        self.saved_kwargs = None

    def save_pretrained(self, output_dir: str, **kwargs) -> None:
        self.saved_state_dict = kwargs.pop("state_dict")
        self.saved_kwargs = kwargs
        self.config.save_pretrained(output_dir)
        self.generation_config.save_pretrained(output_dir)
        Path(output_dir, "model.safetensors").write_bytes(b"weights")


class AcceleratorStub:
    def __init__(self, unwrapped_model):
        self.unwrapped_model = unwrapped_model

    def unwrap_model(self, model, keep_torch_compile=False):
        del model, keep_torch_compile
        return self.unwrapped_model


def test_save_hf_pretrained_model_delegates_to_hf_and_copies_custom_code(
    tmp_path: Path,
):
    source_dir = tmp_path / "base"
    export_dir = tmp_path / "export"
    source_dir.mkdir()
    (source_dir / "configuration_lfm2_moe.py").write_text("# config code\n")
    (source_dir / "modeling_lfm2_moe.py").write_text("# modeling code\n")
    model = ModelStub(source_dir)
    state_dict = {"model.embed_tokens.weight": torch.ones(2, 3)}

    _save_hf_pretrained_model(
        model_to_save=model,
        state_dict=state_dict,
        export_dir=str(export_dir),
    )

    config = json.loads((export_dir / "config.json").read_text())
    generation_config = json.loads((export_dir / "generation_config.json").read_text())

    assert model.saved_state_dict is state_dict
    assert model.saved_kwargs["is_main_process"] is True
    assert model.saved_kwargs["max_shard_size"] == HF_EXPORT_MAX_SHARD_SIZE
    assert config["model_type"] == "lfm2_moe"
    assert config["max_position_embeddings"] == 131072
    assert generation_config["eos_token_id"] == 7
    assert (export_dir / "model.safetensors").read_bytes() == b"weights"
    assert (export_dir / "configuration_lfm2_moe.py").read_text() == "# config code\n"
    assert (export_dir / "modeling_lfm2_moe.py").read_text() == "# modeling code\n"


def test_canonicalize_hf_export_state_dict_unpacks_lfm2_moe_experts():
    gate_up = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    down = torch.arange(100, 100 + 2 * 3 * 2).reshape(2, 3, 2)

    canonical = _canonicalize_hf_export_state_dict(
        {
            "model.layers.2.feed_forward.experts.gate_up_proj": gate_up,
            "model.layers.2.feed_forward.experts.down_proj": down,
            "model.layers.2.feed_forward.gate.weight": torch.ones(2, 3),
        }
    )

    assert "model.layers.2.feed_forward.experts.gate_up_proj" not in canonical
    assert "model.layers.2.feed_forward.experts.down_proj" not in canonical
    assert torch.equal(
        canonical["model.layers.2.feed_forward.experts.0.w1.weight"],
        gate_up[0, :2],
    )
    assert torch.equal(
        canonical["model.layers.2.feed_forward.experts.0.w3.weight"],
        gate_up[0, 2:],
    )
    assert torch.equal(
        canonical["model.layers.2.feed_forward.experts.1.w2.weight"],
        down[1],
    )
    assert torch.equal(
        canonical["model.layers.2.feed_forward.gate.weight"],
        torch.ones(2, 3),
    )


def test_root_hf_export_uses_full_state_dict_and_hf_save_pretrained(
    tmp_path: Path,
    monkeypatch,
):
    source_dir = tmp_path / "base"
    source_dir.mkdir()
    (source_dir / "configuration_lfm2_moe.py").write_text("# config code\n")
    (source_dir / "modeling_lfm2_moe.py").write_text("# modeling code\n")
    unwrapped_model = ModelStub(source_dir)
    gate_up = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    down = torch.arange(100, 100 + 2 * 3 * 2).reshape(2, 3, 2)
    captured_options = []

    def fake_get_model_state_dict(model, *, options):
        captured_options.append(options)
        return {
            "model.layers.2.feed_forward.experts.gate_up_proj": gate_up,
            "model.layers.2.feed_forward.experts.down_proj": down,
        }

    monkeypatch.setattr(
        "leap_finetune.utils.model_utils.get_model_state_dict",
        fake_get_model_state_dict,
    )
    monkeypatch.setattr("leap_finetune.utils.model_utils._world_barrier", lambda: None)

    _save_root_hf_export(
        model=object(),
        accelerator=AcceleratorStub(unwrapped_model),
        output_dir=str(tmp_path / "export"),
        processing_class=None,
        data_collator=None,
        training_args={"arg": "value"},
        staging_dir=None,
    )

    assert len(captured_options) == 1
    assert captured_options[0].full_state_dict is True
    assert captured_options[0].cpu_offload is True
    assert "model.layers.2.feed_forward.experts.gate_up_proj" not in (
        unwrapped_model.saved_state_dict
    )
    assert torch.equal(
        unwrapped_model.saved_state_dict[
            "model.layers.2.feed_forward.experts.0.w1.weight"
        ],
        gate_up[0, :2],
    )
    assert torch.equal(
        unwrapped_model.saved_state_dict[
            "model.layers.2.feed_forward.experts.1.w2.weight"
        ],
        down[1],
    )
