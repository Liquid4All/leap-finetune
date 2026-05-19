from __future__ import annotations

import json
from pathlib import Path

import torch

from leap_finetune.utils.model_utils import (
    HF_EXPORT_MAX_SHARD_SIZE,
    MANUAL_SHARDED_FORMAT_VERSION,
    _canonicalize_hf_export_state_dict,
    _save_hf_pretrained_model,
    _save_root_metadata,
    _save_root_hf_export,
    build_manual_sharded_export_metadata_from_config,
    finalize_manual_sharded_export_metadata,
    load_manual_sharded_checkpoint_metadata,
)
from leap_finetune.utils.load_models import normalize_model_config_overrides


class ConfigStub:
    auto_map = {
        "AutoConfig": "configuration_lfm2_moe.Lfm2MoeConfig",
        "AutoModelForCausalLM": "modeling_lfm2_moe.Lfm2MoeForCausalLM",
    }

    def __init__(self, source_dir: Path):
        self._name_or_path = str(source_dir)
        self.model_type = "lfm2_moe"
        self.max_position_embeddings = 131072
        self.rope_parameters = {"rope_type": "default", "rope_theta": 1000000.0}
        self.rope_scaling = None
        self.rope_theta = 1000000.0
        self.default_theta = 1000000.0
        self.tie_word_embeddings = True

    def save_pretrained(self, output_dir: str) -> None:
        config = {
            "model_type": self.model_type,
            "auto_map": self.auto_map,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_parameters": self.rope_parameters,
            "rope_theta": self.rope_theta,
            "tie_word_embeddings": self.tie_word_embeddings,
        }
        if self.rope_scaling is not None:
            config["rope_scaling"] = self.rope_scaling
        Path(output_dir, "config.json").write_text(json.dumps(config))


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


class AliasedRopeConfigStub:
    auto_map = {}

    def __init__(self):
        self.model_type = "lfm2_moe"
        self.max_position_embeddings = 128000
        self.rope_parameters = {"rope_type": "default", "rope_theta": 1000000.0}
        self.default_theta = 1000000.0
        self.tie_word_embeddings = False

    @property
    def rope_scaling(self):
        return self.rope_parameters

    @rope_scaling.setter
    def rope_scaling(self, value):
        self.rope_parameters = value

    def save_pretrained(self, output_dir: str) -> None:
        config = {
            "model_type": self.model_type,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_parameters": self.rope_parameters,
            "tie_word_embeddings": self.tie_word_embeddings,
        }
        if "rope_scaling" in self.__dict__:
            config["rope_scaling"] = self.__dict__["rope_scaling"]
        Path(output_dir, "config.json").write_text(json.dumps(config))


class AliasedRopeModelStub(ModelStub):
    def __init__(self):
        self.config = AliasedRopeConfigStub()
        self.generation_config = GenerationConfigStub()
        self.saved_state_dict = None
        self.saved_kwargs = None


class AcceleratorStub:
    def __init__(self, unwrapped_model):
        self.unwrapped_model = unwrapped_model

    def unwrap_model(self, model, keep_torch_compile=False):
        del model, keep_torch_compile
        return self.unwrapped_model


class TokenizerStub:
    chat_template = "template-v1"


def test_manual_sharded_metadata_preserves_run_config_and_active_template():
    training_config = {
        "model_name": "LiquidAI/LFM2-24B-A2B",
        "model_config": {
            "rope_scaling": {"rope_type": "yarn", "factor": 10.0},
        },
        "train_config": {
            "max_length": 327680,
            "chat_template_path": "job_configs/templates/lfm.jinja",
        },
    }

    metadata = build_manual_sharded_export_metadata_from_config(
        training_config,
        processing_class=TokenizerStub(),
    )

    assert metadata["training_config"] == training_config
    assert metadata["base_model_name"] == "LiquidAI/LFM2-24B-A2B"
    assert metadata["model_config"] == training_config["model_config"]
    assert metadata["max_length"] == 327680
    assert metadata["chat_template"] == "template-v1"
    assert len(metadata["chat_template_sha256"]) == 64


def test_root_metadata_records_checkpoint_contract(tmp_path: Path):
    _save_root_metadata(
        str(tmp_path),
        save_only_model=False,
        checkpoint_format="both",
        export_metadata={
            "base_model_name": "LiquidAI/LFM2-24B-A2B",
            "max_length": 327680,
        },
    )

    metadata = load_manual_sharded_checkpoint_metadata(str(tmp_path))
    assert metadata["format_version"] == MANUAL_SHARDED_FORMAT_VERSION
    assert metadata["checkpoint_format"] == "both"
    assert metadata["has_resume_state"] is True
    assert metadata["has_optimizer_state"] is True
    assert metadata["base_model_name"] == "LiquidAI/LFM2-24B-A2B"
    assert metadata["max_length"] == 327680


def test_finalize_manual_sharded_metadata_uses_current_template():
    metadata = finalize_manual_sharded_export_metadata(
        {"chat_template": "stale", "base_model_name": "base"},
        processing_class=TokenizerStub(),
    )

    assert metadata["base_model_name"] == "base"
    assert metadata["chat_template"] == "template-v1"
    assert len(metadata["chat_template_sha256"]) == 64


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
    assert config["tie_word_embeddings"] is True
    assert generation_config["eos_token_id"] == 7
    assert (export_dir / "model.safetensors").read_bytes() == b"weights"
    assert (export_dir / "configuration_lfm2_moe.py").read_text() == "# config code\n"
    assert (export_dir / "modeling_lfm2_moe.py").read_text() == "# modeling code\n"


def test_save_hf_pretrained_model_preserves_yarn_and_trained_lm_head(
    tmp_path: Path,
):
    source_dir = tmp_path / "base"
    export_dir = tmp_path / "export"
    source_dir.mkdir()
    model = ModelStub(source_dir)
    state_dict = {
        "model.embed_tokens.weight": torch.ones(2, 3),
        "lm_head.weight": torch.ones(2, 3),
    }

    _save_hf_pretrained_model(
        model_to_save=model,
        state_dict=state_dict,
        export_dir=str(export_dir),
        export_metadata={
            "model_config": {
                "rope_scaling": {
                    "rope_type": "yarn",
                    "factor": 10.0,
                    "original_max_position_embeddings": 128000,
                },
                "max_position_embeddings": 327680,
            },
            "max_length": 327680,
        },
    )

    config = json.loads((export_dir / "config.json").read_text())
    assert config["max_position_embeddings"] == 327680
    assert config["rope_parameters"]["rope_type"] == "yarn"
    assert config["rope_parameters"]["factor"] == 10.0
    assert config["rope_parameters"]["original_max_position_embeddings"] == 128000
    assert config["rope_parameters"]["rope_theta"] == 1000000.0
    assert config["rope_scaling"]["type"] == "yarn"
    assert config["rope_scaling"]["factor"] == 10.0
    assert config["rope_scaling"]["original_max_position_embeddings"] == 128000
    assert config["tie_word_embeddings"] is False


def test_lfm2_rope_theta_override_is_mirrored_to_rope_parameters():
    config = ConfigStub(Path("base"))

    normalized = normalize_model_config_overrides(
        config,
        {"rope_theta": 5000000.0},
    )

    assert normalized["rope_theta"] == 5000000.0
    assert normalized["rope_parameters"]["rope_theta"] == 5000000.0
    assert normalized["rope_parameters"]["rope_type"] == "default"


def test_save_hf_pretrained_model_preserves_rope_theta_without_yarn(
    tmp_path: Path,
):
    source_dir = tmp_path / "base"
    export_dir = tmp_path / "export"
    source_dir.mkdir()
    model = ModelStub(source_dir)

    _save_hf_pretrained_model(
        model_to_save=model,
        state_dict={"model.embed_tokens.weight": torch.ones(2, 3)},
        export_dir=str(export_dir),
        export_metadata={
            "model_config": {
                "rope_theta": 5000000.0,
                "max_position_embeddings": 327680,
            },
            "max_length": 327680,
        },
    )

    config = json.loads((export_dir / "config.json").read_text())
    assert config["max_position_embeddings"] == 327680
    assert config["rope_theta"] == 5000000.0
    assert config["rope_parameters"]["rope_theta"] == 5000000.0
    assert config["rope_parameters"]["rope_type"] == "default"


def test_save_hf_pretrained_model_writes_literal_rope_scaling_key_for_alias_config(
    tmp_path: Path,
):
    export_dir = tmp_path / "export"
    model = AliasedRopeModelStub()

    _save_hf_pretrained_model(
        model_to_save=model,
        state_dict={"model.embed_tokens.weight": torch.ones(2, 3)},
        export_dir=str(export_dir),
        export_metadata={
            "model_config": {
                "rope_scaling": {
                    "rope_type": "yarn",
                    "factor": 10.0,
                    "original_max_position_embeddings": 128000,
                },
            },
            "max_length": 327680,
        },
    )

    config = json.loads((export_dir / "config.json").read_text())
    assert config["rope_parameters"]["rope_type"] == "yarn"
    assert config["rope_scaling"]["type"] == "yarn"
    assert config["rope_scaling"]["rope_type"] == "yarn"
    assert config["rope_scaling"]["factor"] == 10.0


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
