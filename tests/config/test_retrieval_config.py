import pytest

from leap_finetune.config.parser import materialize_job_config, parse_job_config

pytestmark = pytest.mark.configs


@pytest.mark.parametrize("dataset_type", ["embedding", "colbert"])
def test_retrieval_config_materializes_defaults(tmp_path, dataset_type):
    data_path = tmp_path / "retrieval.jsonl"
    data_path.write_text(
        '{"query":"q","positive":"p","negative":"n"}\n',
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
project_name: retrieval-test
model_name: LiquidAI/LFM2.5-Embedding-350M
training_type: {dataset_type}
dataset:
  path: {data_path}
  type: {dataset_type}
  test_size: 0.5
training_config: {{}}
peft_config:
  use_peft: false
""",
        encoding="utf-8",
    )
    job = materialize_job_config(parse_job_config(config_path))
    assert job.training_type == dataset_type
    assert job.dataset.dataset_type == dataset_type
    assert job.training_config.value["gather_across_devices"] is True
    assert job.training_config.value["gradient_accumulation_steps"] == 1


def test_retrieval_config_rejects_type_mismatch(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
project_name: mismatch
model_name: LiquidAI/LFM2.5-Embedding-350M
training_type: embedding
dataset:
  path: dataset.jsonl
  type: colbert
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="matching training_type"):
        parse_job_config(config_path)


def test_retrieval_config_rejects_peft(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
project_name: peft
model_name: LiquidAI/LFM2.5-Embedding-350M
training_type: embedding
dataset:
  path: dataset.jsonl
  type: embedding
peft_config:
  use_peft: true
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="PEFT is not yet supported"):
        parse_job_config(config_path)
