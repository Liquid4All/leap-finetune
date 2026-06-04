"""Unit tests for async eval: config parsing, callback dispatch,
backend abstraction (FakeBackend), sidecar marker lifecycle.

GPU-backed integration tests live separately and are slurm-only.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.configs


# === AsyncEvalConfig parsing ===


class TestAsyncEvalConfig:
    def test_default_is_sync(self):
        from leap_finetune.evaluation.async_eval_config import AsyncEvalConfig

        cfg = AsyncEvalConfig.from_dict(None)
        assert cfg.mode == "sync"

        cfg = AsyncEvalConfig.from_dict({})
        assert cfg.mode == "sync"

    def test_sidecar_defaults(self):
        from leap_finetune.evaluation.async_eval_config import AsyncEvalConfig

        cfg = AsyncEvalConfig.from_dict({"mode": "sidecar"})
        assert cfg.mode == "sidecar"
        assert cfg.vllm_gpus == 1
        assert cfg.tensor_parallel_size == 1
        assert cfg.sbatch.time is None  # no default cap; inherits partition default
        assert cfg.on_overlap == "skip"

    def test_reserved_defaults(self):
        from leap_finetune.evaluation.async_eval_config import AsyncEvalConfig

        cfg = AsyncEvalConfig.from_dict({"mode": "reserved", "vllm_gpus": 2})
        assert cfg.mode == "reserved"
        assert cfg.vllm_gpus == 2
        assert cfg.reserved.weight_reload == "respawn"
        assert cfg.reserved.server_port == 8100

    def test_rejects_bogus_mode(self):
        from leap_finetune.evaluation.async_eval_config import AsyncEvalConfig

        with pytest.raises(ValueError, match="async_eval.mode"):
            AsyncEvalConfig.from_dict({"mode": "bogus"})

    def test_rejects_tp_exceeds_vllm_gpus(self):
        from leap_finetune.evaluation.async_eval_config import AsyncEvalConfig

        with pytest.raises(ValueError, match="tensor_parallel_size"):
            AsyncEvalConfig.from_dict(
                {"mode": "sidecar", "vllm_gpus": 1, "tensor_parallel_size": 4}
            )

    def test_rejects_invalid_overlap(self):
        from leap_finetune.evaluation.async_eval_config import AsyncEvalConfig

        with pytest.raises(ValueError, match="on_overlap"):
            AsyncEvalConfig.from_dict({"mode": "sidecar", "on_overlap": "bogus"})

    def test_rejects_invalid_weight_reload(self):
        from leap_finetune.evaluation.async_eval_config import AsyncEvalConfig

        with pytest.raises(ValueError, match="weight_reload"):
            AsyncEvalConfig.from_dict(
                {"mode": "reserved", "reserved": {"weight_reload": "bogus"}}
            )

    def test_round_trip(self):
        from leap_finetune.evaluation.async_eval_config import AsyncEvalConfig

        cfg = AsyncEvalConfig.from_dict(
            {
                "mode": "sidecar",
                "vllm_gpus": 2,
                "tensor_parallel_size": 2,
                "gpu_memory_utilization": 0.85,
                "sbatch": {"partition": "defq", "time": "01:00:00"},
                "failure": {"max_consecutive": 5},
            }
        )
        d = cfg.to_dict()
        cfg2 = AsyncEvalConfig.from_dict(d)
        assert cfg2.mode == "sidecar"
        assert cfg2.vllm_gpus == 2
        assert cfg2.tensor_parallel_size == 2
        assert cfg2.gpu_memory_utilization == 0.85
        assert cfg2.sbatch.partition == "defq"
        assert cfg2.sbatch.time == "01:00:00"
        assert cfg2.failure.max_consecutive == 5


# === Dispatch helper ===


class TestMakeEvalCallback:
    def test_sync_returns_benchmark_eval_callback(self):
        from leap_finetune.evaluation import BenchmarkEvalCallback, make_eval_callback

        cb = make_eval_callback(benchmarks=[], async_eval_cfg=None)
        assert isinstance(cb, BenchmarkEvalCallback)

    def test_sync_explicit(self):
        from leap_finetune.evaluation import BenchmarkEvalCallback, make_eval_callback

        cb = make_eval_callback(benchmarks=[], async_eval_cfg={"mode": "sync"})
        assert isinstance(cb, BenchmarkEvalCallback)

    def test_sidecar_returns_sidecar_callback(self, tmp_path):
        from leap_finetune.evaluation import make_eval_callback
        from leap_finetune.evaluation.sidecar_callback import SidecarEvalCallback

        cb = make_eval_callback(
            benchmarks=[],
            async_eval_cfg={"mode": "sidecar"},
            output_dir=str(tmp_path),
        )
        assert isinstance(cb, SidecarEvalCallback)

    def test_reserved_requires_server_url(self):
        from leap_finetune.evaluation import make_eval_callback

        with pytest.raises(RuntimeError, match="server_url"):
            make_eval_callback(
                benchmarks=[],
                async_eval_cfg={"mode": "reserved"},
                server_url=None,
            )

    def test_reserved_returns_reserved_callback(self, tmp_path):
        from leap_finetune.evaluation import make_eval_callback
        from leap_finetune.evaluation.reserved_callback import ReservedEvalCallback

        cb = make_eval_callback(
            benchmarks=[],
            async_eval_cfg={"mode": "reserved"},
            server_url="http://localhost:8100",
            output_dir=str(tmp_path),
            eval_gpu_ids="0",
        )
        assert isinstance(cb, ReservedEvalCallback)


# === Backend abstraction (FakeBackend → benchmark dispatch) ===


class _FakeBackend:
    """Returns canned generation/logprob results so we can test
    ``Benchmark.evaluate_with_backend`` without a real model."""

    name = "fake"

    def __init__(self, generations=None, logprobs=None):
        self._generations = generations or []
        self._logprobs = logprobs or []

    def generate(self, requests):
        from leap_finetune.evaluation.backend import GenerateResult

        out = []
        for i, _ in enumerate(requests):
            text = self._generations[i] if i < len(self._generations) else ""
            out.append(GenerateResult(text=text))
        return out

    def logprobs(self, requests):
        from leap_finetune.evaluation.backend import LogprobResult

        out = []
        for i, _ in enumerate(requests):
            scores = self._logprobs[i] if i < len(self._logprobs) else []
            out.append(LogprobResult(logprobs=scores))
        return out

    def close(self):
        pass


class TestEvaluateWithBackend:
    def test_llm_generation_short_answer(self):
        from leap_finetune.evaluation.llm_benchmarks import LLMGenerationBenchmark

        bench = LLMGenerationBenchmark(
            name="t",
            path="UNUSED",
            tokenizer=None,  # not used by evaluate_with_backend
            metric="short_answer",
            max_new_tokens=8,
        )

        samples = [
            {
                "id": "a",
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                ],
            },
            {
                "id": "b",
                "messages": [
                    {"role": "user", "content": "What is 3+3?"},
                    {"role": "assistant", "content": "6"},
                ],
            },
        ]
        backend = _FakeBackend(generations=["the answer is 4", "no idea"])
        result = bench.evaluate_with_backend(backend, samples)
        # First sample: "4" is contained in "the answer is 4" → 1.0
        # Second sample: "6" not in "no idea" → 0.0
        assert result.count == 2
        assert result.metrics["score"] == pytest.approx(1.0)

    def test_llm_logprob_picks_argmax(self):
        from leap_finetune.evaluation.llm_benchmarks import LLMLogprobBenchmark

        bench = LLMLogprobBenchmark(
            name="t",
            path="UNUSED",
            tokenizer=None,
        )
        samples = [
            {
                "id": "a",
                "messages": [{"role": "user", "content": "Q?"}],
                "options": ["x", "y", "z"],
                "answer_id": 1,
            },
            {
                "id": "b",
                "messages": [{"role": "user", "content": "Q?"}],
                "options": ["x", "y"],
                "answer_id": 0,
            },
        ]
        # First: argmax of [0.1, 0.5, 0.2] = 1 → matches answer_id=1 → 1.0
        # Second: argmax of [0.1, 0.2] = 1 → does not match answer_id=0 → 0.0
        backend = _FakeBackend(logprobs=[[0.1, 0.5, 0.2], [0.1, 0.2]])
        result = bench.evaluate_with_backend(backend, samples)
        assert result.count == 2
        assert result.metrics["score"] == pytest.approx(1.0)

    def test_default_evaluate_with_backend_raises(self):
        from leap_finetune.evaluation.base import Benchmark

        class Custom(Benchmark):
            def load_samples(self):
                return []

            def score_sample(self, model, sample, device):
                return 0.0

        c = Custom("custom")
        with pytest.raises(NotImplementedError, match="evaluate_with_backend"):
            c.evaluate_with_backend(_FakeBackend(), [{"id": "x"}])


# === Sidecar callback marker lifecycle (subprocess.run mocked) ===


class TestSidecarCallbackMarker:
    def test_skip_when_marker_exists(self, tmp_path):
        from leap_finetune.evaluation.async_eval_config import AsyncEvalConfig
        from leap_finetune.evaluation.sidecar_callback import SidecarEvalCallback

        eval_dir = tmp_path / "_async_eval"
        eval_dir.mkdir()
        (eval_dir / ".in_flight").write_text("999")

        cb = SidecarEvalCallback(
            benchmarks=[MagicMock(name="bench1")],  # non-empty so callback proceeds
            cfg=AsyncEvalConfig.from_dict({"mode": "sidecar", "on_overlap": "skip"}),
            benchmark_configs={"benchmarks": [{"name": "x"}]},
            output_dir=str(tmp_path),
            wandb_run_id=None,
        )

        # Call on_evaluate; should skip without invoking sbatch
        with patch.object(cb, "_submit") as submit_mock:
            with patch(
                "leap_finetune.evaluation.sidecar_callback.is_rank_zero",
                return_value=True,
            ):
                state = MagicMock()
                state.global_step = 100
                cb.on_evaluate(MagicMock(), state, MagicMock(), model=MagicMock())
            assert not submit_mock.called

    def test_failure_disables_after_max_consecutive(self, tmp_path):
        from leap_finetune.evaluation.async_eval_config import AsyncEvalConfig
        from leap_finetune.evaluation.sidecar_callback import SidecarEvalCallback

        cb = SidecarEvalCallback(
            benchmarks=[MagicMock(name="bench1")],
            cfg=AsyncEvalConfig.from_dict(
                {"mode": "sidecar", "failure": {"max_consecutive": 2}}
            ),
            benchmark_configs={"benchmarks": [{"name": "x"}]},
            output_dir=str(tmp_path),
            wandb_run_id=None,
        )

        with patch(
            "leap_finetune.evaluation.sidecar_callback.is_rank_zero",
            return_value=True,
        ):
            with patch.object(cb, "_submit", side_effect=RuntimeError("boom")):
                state = MagicMock()
                control = MagicMock(should_evaluate=True)
                state.global_step = 1
                cb.on_step_end(MagicMock(), state, control, model=MagicMock())
                state.global_step = 2
                cb.on_step_end(MagicMock(), state, control, model=MagicMock())

        assert cb._disabled


# === Sbatch script rendering ===


class TestSbatchTemplate:
    def test_renders_runnable_script(self, tmp_path):
        from leap_finetune.evaluation.sbatch_template import render_sbatch_script

        sub = render_sbatch_script(
            output_dir=tmp_path,
            trigger_step=42,
            checkpoint_path=tmp_path / "ckpt",
            benchmark_configs_json=tmp_path / "bench.json",
            modality="text",
            wandb_run_id="abc123",
            wandb_project="my-project",
            job_name="leap_eval_step_42",
            vllm_gpus=1,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            dtype="bfloat16",
            max_model_len=None,
            sbatch_partition="defq",
            sbatch_account=None,
            sbatch_time="00:15:00",
            sbatch_extra_args=["--qos=high"],
        )
        assert sub.script_path.exists()
        content = sub.script_path.read_text()
        assert "#SBATCH --job-name=leap_eval_step_42" in content
        assert "#SBATCH --partition=defq" in content
        assert "#SBATCH --time=00:15:00" in content
        assert "#SBATCH --qos=high" in content
        assert "trap 'rm -f" in content
        assert "leap_finetune.evaluation.async_runner_main" in content
        # Args are split across lines with shlex-quoted separation
        assert "--trigger-step" in content and "42" in content
        assert "--wandb-run-id" in content and "abc123" in content
        # Script should be executable
        assert sub.script_path.stat().st_mode & 0o111


# === JobConfig + config_parser passthrough ===


class TestJobConfigAsyncEval:
    def test_async_eval_threaded_through_to_dict(self):
        from leap_finetune.training_configs.job_config import JobConfig

        jc = JobConfig(
            job_name="t",
            async_eval={"mode": "sidecar", "vllm_gpus": 2},
        )
        d = jc.to_dict()
        assert d["async_eval"] == {"mode": "sidecar", "vllm_gpus": 2}

    def test_config_parser_validates_async_eval_block(self, tmp_path):
        import yaml

        from leap_finetune.utils.config_parser import parse_job_config

        cfg = {
            "project_name": "test",
            "training_type": "sft",
            "model_name": "LFM2-1.2B",
            "dataset": {"path": "fake/path", "type": "sft"},
            "training_config": {"extends": "DEFAULT_SFT"},
            "async_eval": {"mode": "bogus"},  # invalid
        }
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg))
        with pytest.raises(ValueError, match="async_eval.mode"):
            parse_job_config(str(p))
