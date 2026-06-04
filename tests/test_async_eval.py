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

    def test_failure_defaults_include_submit_retry(self):
        from leap_finetune.evaluation.async_eval_config import AsyncEvalConfig

        cfg = AsyncEvalConfig.from_dict({"mode": "sidecar"})
        assert cfg.failure.max_submit_attempts == 3
        assert cfg.failure.submit_retry_backoff == 2.0

    def test_failure_partial_override_keeps_defaults(self):
        from leap_finetune.evaluation.async_eval_config import AsyncEvalConfig

        cfg = AsyncEvalConfig.from_dict(
            {"mode": "sidecar", "failure": {"max_consecutive": 5}}
        )
        assert cfg.failure.max_consecutive == 5
        assert cfg.failure.max_submit_attempts == 3
        assert cfg.failure.submit_retry_backoff == 2.0

    def test_round_trip_preserves_submit_retry(self):
        from leap_finetune.evaluation.async_eval_config import AsyncEvalConfig

        cfg = AsyncEvalConfig.from_dict(
            {
                "mode": "sidecar",
                "failure": {
                    "max_consecutive": 2,
                    "max_submit_attempts": 7,
                    "submit_retry_backoff": 0.5,
                },
            }
        )
        cfg2 = AsyncEvalConfig.from_dict(cfg.to_dict())
        assert cfg2.failure.max_consecutive == 2
        assert cfg2.failure.max_submit_attempts == 7
        assert cfg2.failure.submit_retry_backoff == 0.5


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

        # Call on_step_end with should_evaluate=True; should skip submission
        # because the .in_flight marker is present (on_overlap=skip).
        with patch.object(cb, "_submit") as submit_mock:
            with patch(
                "leap_finetune.evaluation.sidecar_callback.is_rank_zero",
                return_value=True,
            ):
                state = MagicMock()
                state.global_step = 100
                control = MagicMock(should_evaluate=True)
                cb.on_step_end(MagicMock(), state, control, model=MagicMock())
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
                control = MagicMock(should_evaluate=True)
                state = MagicMock()
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


# === Sbatch retry loop in _submit ===


class _FakeCompletedProcess:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _make_sidecar_for_submit(
    tmp_path, *, failure_overrides=None, with_benchmarks=False
):
    """Build a SidecarEvalCallback whose dependencies are stubbed enough that
    ``_submit`` can be exercised against a mocked ``subprocess.run``."""
    from leap_finetune.evaluation.async_eval_config import AsyncEvalConfig
    from leap_finetune.evaluation.sidecar_callback import SidecarEvalCallback

    failure = {
        "max_consecutive": 99,
        "max_submit_attempts": 3,
        "submit_retry_backoff": 1.0,
    }
    if failure_overrides:
        failure.update(failure_overrides)

    benches = [MagicMock(name="bench1")] if with_benchmarks else []
    cb = SidecarEvalCallback(
        benchmarks=benches,
        cfg=AsyncEvalConfig.from_dict({"mode": "sidecar", "failure": failure}),
        benchmark_configs={"benchmarks": []},
        output_dir=str(tmp_path),
        wandb_run_id=None,
    )
    return cb


def _make_model_mock():
    """MagicMock model whose ``save_pretrained`` actually creates the
    checkpoint dir so the downstream ``write_text`` in ``_submit`` works.

    ``.module`` is removed so the ``hasattr(model, "module")`` branch in
    ``_submit`` falls through to ``model`` itself (the one with our
    side_effect). Without this, ``model.module`` is a fresh MagicMock and
    the dir never gets created.
    """
    from pathlib import Path as _P

    model = MagicMock()
    del model.module  # AttributeError on access; hasattr -> False

    def _save(path, *a, **kw):
        _P(path).mkdir(parents=True, exist_ok=True)

    model.save_pretrained.side_effect = _save
    return model


def _patch_submit_prereqs(monkeypatch, *, ckpt_root):
    """Stub the parts of _submit that touch external code (render the
    sbatch script + scrub env) so the retry loop is the only path the
    test exercises."""
    import leap_finetune.evaluation.sidecar_callback as sc

    monkeypatch.setattr(
        sc,
        "render_sbatch_script",
        lambda **kw: MagicMock(
            script_path=ckpt_root / "fake.sh",
        ),
    )
    monkeypatch.setattr(sc, "_clean_subprocess_env", lambda: {})


class TestSidecarSubmitRetry:
    def test_succeeds_first_attempt_no_sleep(self, tmp_path, monkeypatch):
        import leap_finetune.evaluation.sidecar_callback as sc

        cb = _make_sidecar_for_submit(tmp_path)
        _patch_submit_prereqs(monkeypatch, ckpt_root=tmp_path)
        run_calls = []
        sleep_calls = []
        monkeypatch.setattr(
            sc, "time", MagicMock(sleep=lambda s: sleep_calls.append(s))
        )
        monkeypatch.setattr(
            sc.subprocess,
            "run",
            lambda *a, **kw: (
                run_calls.append(1)
                or _FakeCompletedProcess(0, "Submitted batch job 123\n")
            ),
        )

        model = _make_model_mock()
        state = MagicMock(global_step=42)
        jobid = cb._submit(model, state, MagicMock())

        assert jobid == "123"
        assert len(run_calls) == 1
        assert sleep_calls == []
        # Marker written after successful submit, format "jobid:step".
        assert (tmp_path / "_async_eval" / ".in_flight").read_text() == "123:42"

    def test_retries_then_succeeds(self, tmp_path, monkeypatch):
        import leap_finetune.evaluation.sidecar_callback as sc

        cb = _make_sidecar_for_submit(tmp_path)
        _patch_submit_prereqs(monkeypatch, ckpt_root=tmp_path)
        sleep_calls = []
        monkeypatch.setattr(
            sc, "time", MagicMock(sleep=lambda s: sleep_calls.append(s))
        )

        results = iter(
            [
                _FakeCompletedProcess(1, stderr="busy"),
                _FakeCompletedProcess(1, stderr="busy"),
                _FakeCompletedProcess(0, "Submitted batch job 999\n"),
            ]
        )
        monkeypatch.setattr(sc.subprocess, "run", lambda *a, **kw: next(results))

        model = _make_model_mock()
        state = MagicMock(global_step=7)
        jobid = cb._submit(model, state, MagicMock())

        assert jobid == "999"
        # backoff=1.0 → sleeps [1.0, 2.0] (2 retries before the 3rd succeeds).
        assert sleep_calls == [1.0, 2.0]
        # Marker written only after the successful (3rd) submit.
        assert (tmp_path / "_async_eval" / ".in_flight").read_text() == "999:7"

    def test_exhausts_attempts_raises_no_marker_left(self, tmp_path, monkeypatch):
        """Bug fix: marker must NOT exist after all retries fail (was being
        written pre-loop and leaking)."""
        import leap_finetune.evaluation.sidecar_callback as sc

        cb = _make_sidecar_for_submit(tmp_path)
        _patch_submit_prereqs(monkeypatch, ckpt_root=tmp_path)
        monkeypatch.setattr(sc, "time", MagicMock(sleep=lambda s: None))
        monkeypatch.setattr(
            sc.subprocess,
            "run",
            lambda *a, **kw: _FakeCompletedProcess(1, stderr="permanent failure"),
        )

        model = _make_model_mock()
        state = MagicMock(global_step=11)
        with pytest.raises(RuntimeError, match="after 3 attempt"):
            cb._submit(model, state, MagicMock())
        assert not (tmp_path / "_async_eval" / ".in_flight").exists()

    def test_filenotfound_fails_fast(self, tmp_path, monkeypatch):
        import leap_finetune.evaluation.sidecar_callback as sc

        cb = _make_sidecar_for_submit(tmp_path)
        _patch_submit_prereqs(monkeypatch, ckpt_root=tmp_path)
        run_calls = []

        def boom(*a, **kw):
            run_calls.append(1)
            raise FileNotFoundError("sbatch")

        monkeypatch.setattr(sc.subprocess, "run", boom)

        model = _make_model_mock()
        state = MagicMock(global_step=1)
        with pytest.raises(RuntimeError, match="sbatch.*not found"):
            cb._submit(model, state, MagicMock())
        # No retry, no marker leak.
        assert len(run_calls) == 1
        assert not (tmp_path / "_async_eval" / ".in_flight").exists()

    def test_zero_backoff_no_sleep(self, tmp_path, monkeypatch):
        import leap_finetune.evaluation.sidecar_callback as sc

        cb = _make_sidecar_for_submit(
            tmp_path, failure_overrides={"submit_retry_backoff": 0.0}
        )
        _patch_submit_prereqs(monkeypatch, ckpt_root=tmp_path)
        sleep_calls = []
        monkeypatch.setattr(
            sc, "time", MagicMock(sleep=lambda s: sleep_calls.append(s))
        )
        results = iter(
            [
                _FakeCompletedProcess(1, stderr="x"),
                _FakeCompletedProcess(0, "Submitted batch job 1\n"),
            ]
        )
        monkeypatch.setattr(sc.subprocess, "run", lambda *a, **kw: next(results))

        model = _make_model_mock()
        state = MagicMock(global_step=1)
        cb._submit(model, state, MagicMock())
        # Computed sleep is 0.0; the loop skips the sleep call entirely.
        assert sleep_calls == []

    def test_fire_counts_retried_submit_as_one_failure(self, tmp_path, monkeypatch):
        """Semantics pin: 3 internal sbatch retries that all fail must count
        as 1 increment of _consecutive_failures, not 3."""

        cb = _make_sidecar_for_submit(
            tmp_path,
            failure_overrides={"max_consecutive": 2, "max_submit_attempts": 3},
            with_benchmarks=True,
        )
        with patch.object(
            cb, "_submit", side_effect=RuntimeError("sbatch failed after 3 attempts")
        ):
            with patch(
                "leap_finetune.evaluation.sidecar_callback.is_rank_zero",
                return_value=True,
            ):
                control = MagicMock(should_evaluate=True)
                state = MagicMock(global_step=1)
                cb.on_step_end(MagicMock(), state, control, model=MagicMock())
                assert cb._consecutive_failures == 1
                assert cb._disabled is False


# === Stale .in_flight marker recovery ===


class TestStaleMarkerRecovery:
    def test_clears_when_sacct_reports_terminal(self, tmp_path, monkeypatch):
        """Marker for a job that sacct says is FAILED must be cleared."""
        import leap_finetune.evaluation.sidecar_callback as sc

        cb = _make_sidecar_for_submit(tmp_path)
        eval_dir = tmp_path / "_async_eval"
        eval_dir.mkdir(exist_ok=True)
        marker = eval_dir / ".in_flight"
        marker.write_text("12345:7")

        monkeypatch.setattr(
            sc.subprocess,
            "run",
            lambda *a, **kw: _FakeCompletedProcess(0, "FAILED\n"),
        )
        cb._clear_marker_if_stale(marker)
        assert not marker.exists()

    def test_preserves_when_sacct_reports_running(self, tmp_path, monkeypatch):
        import leap_finetune.evaluation.sidecar_callback as sc

        cb = _make_sidecar_for_submit(tmp_path)
        eval_dir = tmp_path / "_async_eval"
        eval_dir.mkdir(exist_ok=True)
        marker = eval_dir / ".in_flight"
        marker.write_text("12345:7")

        monkeypatch.setattr(
            sc.subprocess,
            "run",
            lambda *a, **kw: _FakeCompletedProcess(0, "RUNNING\n"),
        )
        cb._clear_marker_if_stale(marker)
        assert marker.exists()

    def test_clears_when_sacct_missing_and_marker_old(self, tmp_path, monkeypatch):
        """sacct unavailable + mtime > 6h → fall back to mtime, clear marker."""
        import os

        import leap_finetune.evaluation.sidecar_callback as sc

        cb = _make_sidecar_for_submit(tmp_path)
        eval_dir = tmp_path / "_async_eval"
        eval_dir.mkdir(exist_ok=True)
        marker = eval_dir / ".in_flight"
        marker.write_text("12345:7")
        # Backdate mtime by 7 hours.
        old = marker.stat().st_mtime - 7 * 3600
        os.utime(marker, (old, old))

        def no_sacct(*a, **kw):
            raise FileNotFoundError("sacct")

        monkeypatch.setattr(sc.subprocess, "run", no_sacct)
        cb._clear_marker_if_stale(marker)
        assert not marker.exists()

    def test_preserves_live_job_even_with_old_marker(self, tmp_path, monkeypatch):
        """Adversarial: marker > 6h old AND sacct says RUNNING. The mtime
        fallback must NOT delete a live sidecar's marker (would cause a
        duplicate submit on the next _fire). Regression test for the
        sacct-success path silently falling through to mtime.
        """
        import os

        import leap_finetune.evaluation.sidecar_callback as sc

        cb = _make_sidecar_for_submit(tmp_path)
        eval_dir = tmp_path / "_async_eval"
        eval_dir.mkdir(exist_ok=True)
        marker = eval_dir / ".in_flight"
        marker.write_text("12345:7")
        old = marker.stat().st_mtime - 7 * 3600
        os.utime(marker, (old, old))

        monkeypatch.setattr(
            sc.subprocess,
            "run",
            lambda *a, **kw: _FakeCompletedProcess(0, "RUNNING\n"),
        )
        cb._clear_marker_if_stale(marker)
        assert marker.exists(), (
            "sacct=RUNNING is authoritative; mtime fallback must not "
            "delete a live job's marker"
        )

    @pytest.mark.parametrize(
        "sacct_state",
        ["BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT", "MYSTERY_STATE"],
    )
    def test_clears_on_unenumerated_terminal_states(
        self, tmp_path, monkeypatch, sacct_state
    ):
        """Slurm terminal states beyond {COMPLETED,FAILED,CANCELLED,...} and
        unknown future states must still clear the marker. Default-to-cleared
        avoids stranding the marker forever when slurm reports something
        we don't explicitly enumerate."""
        import leap_finetune.evaluation.sidecar_callback as sc

        cb = _make_sidecar_for_submit(tmp_path)
        eval_dir = tmp_path / "_async_eval"
        eval_dir.mkdir(exist_ok=True)
        marker = eval_dir / ".in_flight"
        marker.write_text("12345:7")

        monkeypatch.setattr(
            sc.subprocess,
            "run",
            lambda *a, **kw: _FakeCompletedProcess(0, f"{sacct_state}\n"),
        )
        cb._clear_marker_if_stale(marker)
        assert not marker.exists(), (
            f"sacct={sacct_state} is non-active; marker must be cleared "
            "even though it's not in the enumerated terminal set"
        )

    def test_keeps_marker_when_any_row_active_in_mixed_output(
        self, tmp_path, monkeypatch
    ):
        """sacct often returns multiple rows (parent + .batch step). If ANY
        row reports an active state, the job is alive — keep the marker
        even if a sibling row shows a non-active state."""
        import leap_finetune.evaluation.sidecar_callback as sc

        cb = _make_sidecar_for_submit(tmp_path)
        eval_dir = tmp_path / "_async_eval"
        eval_dir.mkdir(exist_ok=True)
        marker = eval_dir / ".in_flight"
        marker.write_text("12345:7")

        monkeypatch.setattr(
            sc.subprocess,
            "run",
            lambda *a, **kw: _FakeCompletedProcess(0, "RUNNING\nCOMPLETED\n"),
        )
        cb._clear_marker_if_stale(marker)
        assert marker.exists()

    def test_strips_slurm_state_suffix(self, tmp_path, monkeypatch):
        """sacct uses a trailing ``+`` to flag non-canonical states; the
        parser must strip it before classifying."""
        import leap_finetune.evaluation.sidecar_callback as sc

        cb = _make_sidecar_for_submit(tmp_path)
        eval_dir = tmp_path / "_async_eval"
        eval_dir.mkdir(exist_ok=True)
        marker = eval_dir / ".in_flight"
        marker.write_text("12345:7")

        monkeypatch.setattr(
            sc.subprocess,
            "run",
            lambda *a, **kw: _FakeCompletedProcess(0, "RUNNING+\n"),
        )
        cb._clear_marker_if_stale(marker)
        assert marker.exists(), "trailing '+' must not defeat active-state matching"


# === wandb.define_metric ordering in async_runner_main ===


class _FakeWandb:
    """Records the order of ``define_metric`` and ``log`` calls so we can
    pin the contract: define before log."""

    def __init__(self):
        self.calls: list[tuple[str, tuple, dict]] = []
        self.run = self

    def init(self, **kw):
        self.calls.append(("init", (), kw))
        return self

    def define_metric(self, *a, **kw):
        self.calls.append(("define_metric", a, kw))

    def log(self, *a, **kw):
        self.calls.append(("log", a, kw))

    def finish(self):
        self.calls.append(("finish", (), {}))

    class Settings:
        def __init__(self, **kw):
            pass


class TestAsyncRunnerWandbAxis:
    def _run(self, fake_wandb, results, monkeypatch):
        import sys

        from leap_finetune.evaluation import async_runner_main as arm

        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
        args = MagicMock(
            wandb_run_id="run-xyz",
            wandb_project=None,
            trigger_step=42,
        )
        arm._log_to_wandb(args, results)

    def test_define_metric_before_log(self, monkeypatch):
        fw = _FakeWandb()
        self._run(
            fw,
            {
                "benchmark/refcoco/score": 0.7,
                "benchmark/gsm8k/score": 0.5,
            },
            monkeypatch,
        )

        # All define_metric must precede the first log.
        first_log = next(i for i, c in enumerate(fw.calls) if c[0] == "log")
        define_indices = [i for i, c in enumerate(fw.calls) if c[0] == "define_metric"]
        assert define_indices, "define_metric was never called"
        assert max(define_indices) < first_log

    def test_define_metric_enumerates_every_benchmark_key(self, monkeypatch):
        fw = _FakeWandb()
        self._run(
            fw,
            {
                "benchmark/refcoco/score": 0.7,
                "benchmark/gsm8k/score": 0.5,
                # Non-benchmark keys are not enumerated explicitly.
                "train/loss": 0.1,
            },
            monkeypatch,
        )
        defined_keys = [c[1][0] for c in fw.calls if c[0] == "define_metric"]
        # benchmark/step + the 2 benchmark/* keys + the trailing glob.
        assert "benchmark/step" in defined_keys
        assert "benchmark/refcoco/score" in defined_keys
        assert "benchmark/gsm8k/score" in defined_keys
        assert "benchmark/*" in defined_keys
        assert "train/loss" not in defined_keys
