from leap_finetune.utils.load_models import (
    _LFM25_DEFAULT_CHAT_TEMPLATE_PATH,
    _resolve_chat_template,
    _resolve_model_id,
)


def test_resolve_model_id_expands_liquidai_short_name():
    assert _resolve_model_id("LFM2-24B-A2B") == "LiquidAI/LFM2-24B-A2B"


def test_resolve_model_id_keeps_qualified_hf_id():
    assert _resolve_model_id("LiquidAI/LFM2-24B-A2B") == "LiquidAI/LFM2-24B-A2B"


def test_resolve_model_id_keeps_other_qualified_hf_id():
    assert _resolve_model_id("some-org/some-model") == "some-org/some-model"


def test_resolve_model_id_keeps_existing_local_dir(tmp_path):
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()

    assert _resolve_model_id(str(model_dir)) == str(model_dir)


def test_resolve_chat_template_defaults_lfm25_models_to_tracked_template():
    resolved = _resolve_chat_template(model_name="LiquidAI/LFM2-24B-A2B")

    assert resolved == _LFM25_DEFAULT_CHAT_TEMPLATE_PATH.read_text()


def test_resolve_chat_template_keeps_lfm2_models_on_tokenizer_default():
    assert _resolve_chat_template(model_name="LiquidAI/LFM2-1.2B") is None


def test_resolve_chat_template_does_not_override_local_checkpoint(tmp_path):
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()

    assert _resolve_chat_template(model_name=str(model_dir)) is None


def test_resolve_chat_template_explicit_override_wins():
    assert (
        _resolve_chat_template(
            chat_template="custom-template",
            model_name="LiquidAI/LFM2-24B-A2B",
        )
        == "custom-template"
    )
