import json
import os

import pytest

from leap_finetune.rl.judge import (
    JUDGE_LLM_CONFIG_ENV,
    JudgeLLM,
    JudgeLLMConfig,
    build_judge_runtime_config,
    export_judge_runtime_config,
    judge_llm_reward,
    judge_needs_local_server,
)

pytestmark = pytest.mark.rl


# === Judge LLM fixtures ===


class FakeTokenizer:
    chat_template = "{messages}"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def batch_decode(self, completion_ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        return [json.dumps({"score": ids[0] / 10}) for ids in completion_ids]


class FakeResponse:
    status_code = 200
    text = "ok"

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.requests = []

    def post(self, url, json, timeout):
        self.requests.append((url, json, timeout))
        return FakeResponse({"completion_ids": [[8] for _ in json["prompts"]]})


# === Judge config ===


def test_judge_needs_local_server_only_without_base_url():
    assert judge_needs_local_server({"judge": {"model": "judge"}})
    assert not judge_needs_local_server(
        {"judge": {"model": "judge", "base_url": "http://judge:8001"}}
    )
    assert not judge_needs_local_server(["length_reward"])


def test_build_judge_runtime_config_uses_default_model_and_base_url():
    config = build_judge_runtime_config(
        {"judge": {"max_tokens": 16}},
        default_model="LFM2-1.2B",
        base_url="http://localhost:8001",
    )

    assert config is not None
    assert config.model == "LFM2-1.2B"
    assert config.base_url == "http://localhost:8001"
    assert config.max_tokens == 16


def test_export_judge_runtime_config_sets_worker_env(monkeypatch):
    monkeypatch.delenv(JUDGE_LLM_CONFIG_ENV, raising=False)
    config = JudgeLLMConfig(model="judge", base_url="http://localhost:8001")

    export_judge_runtime_config(config)

    raw = json.loads(os.environ[JUDGE_LLM_CONFIG_ENV])
    assert raw["model"] == "judge"
    assert raw["base_url"] == "http://localhost:8001"


# === Judge scoring ===


def test_judge_scores_completions_with_vllm_server_protocol():
    session = FakeSession()
    judge = JudgeLLM(
        JudgeLLMConfig(
            model="judge",
            base_url="http://localhost:8001",
            max_score=1.0,
        ),
        session=session,
        tokenizer=FakeTokenizer(),
    )

    scores = judge.score(
        [[{"role": "assistant", "content": "42"}]],
        prompts=["What is 40+2?"],
        solution=["42"],
    )

    assert scores == [0.8]
    url, payload, timeout = session.requests[0]
    assert url == "http://localhost:8001/generate/"
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 32
    assert timeout == 120.0
    assert "What is 40+2?" in payload["prompts"][0]
    assert "42" in payload["prompts"][0]


def test_judge_score_parser_prefers_score_field():
    judge = JudgeLLM(
        JudgeLLMConfig(model="judge", base_url="http://localhost:8001"),
        tokenizer=FakeTokenizer(),
    )

    assert judge._parse_score("range 0 to 1, score: 0.75") == 0.75


def test_judge_reward_reads_exported_env(monkeypatch):
    raw_config = JudgeLLMConfig(
        model="judge",
        base_url="http://localhost:8001",
    ).to_json()
    monkeypatch.setenv(JUDGE_LLM_CONFIG_ENV, raw_config)

    import leap_finetune.rl.judge as judge_module

    fake = JudgeLLM(
        JudgeLLMConfig(model="judge", base_url="http://localhost:8001"),
        session=FakeSession(),
        tokenizer=FakeTokenizer(),
    )
    monkeypatch.setattr(judge_module, "_JUDGE_CACHE", fake)
    monkeypatch.setattr(judge_module, "_JUDGE_CACHE_RAW", raw_config)

    scores = judge_llm_reward(["answer"], prompts=["question"], solution=["answer"])

    assert scores == [0.8]
