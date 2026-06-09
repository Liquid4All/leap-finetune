from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.evaluation

# === Async Eval Contracts ===


class _FakeBackend:
    name = "fake"

    def __init__(self, generations=None, logprobs=None):
        self._generations = generations or []
        self._logprobs = logprobs or []

    def generate(self, requests):
        from leap_finetune.evaluation.backend import GenerateResult

        return [
            GenerateResult(
                text=self._generations[i] if i < len(self._generations) else ""
            )
            for i, _ in enumerate(requests)
        ]

    def logprobs(self, requests):
        from leap_finetune.evaluation.backend import LogprobResult

        return [
            LogprobResult(logprobs=self._logprobs[i] if i < len(self._logprobs) else [])
            for i, _ in enumerate(requests)
        ]

    def close(self):
        pass


class _FakeCompletedProcess:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _FakeWandb:
    def __init__(self):
        self.calls: list = []
        self.run = self

    def init(self, **kw):
        self.calls.append(("init",))

    def define_metric(self, *args, **kw):
        self.calls.append(("define_metric", args))

    def log(self, *args, **kw):
        self.calls.append(("log",))

    def finish(self):
        self.calls.append(("finish",))

    class Settings:
        def __init__(self, **kw):
            pass


def _make_sidecar(tmp_path, *, failure_overrides=None, with_benchmarks=False):
    from leap_finetune.evaluation.async_eval_config import AsyncEvalConfig
    from leap_finetune.evaluation.sidecar_callback import SidecarEvalCallback

    failure = {
        "max_consecutive": 99,
        "max_submit_attempts": 3,
        "submit_retry_backoff": 1.0,
    }
    if failure_overrides:
        failure.update(failure_overrides)

    return SidecarEvalCallback(
        benchmarks=[MagicMock(name="bench1")] if with_benchmarks else [],
        cfg=AsyncEvalConfig.from_dict({"mode": "sidecar", "failure": failure}),
        benchmark_configs={"benchmarks": []},
        output_dir=str(tmp_path),
        wandb_run_id=None,
    )


def _make_model_mock():
    model = MagicMock()
    del model.module

    def _save(path, *args, **kwargs):
        Path(path).mkdir(parents=True, exist_ok=True)

    model.save_pretrained.side_effect = _save
    return model


def _patch_submit_prereqs(monkeypatch, *, ckpt_root):
    import leap_finetune.evaluation.sidecar_callback as sc

    monkeypatch.setattr(
        sc,
        "render_sbatch_script",
        lambda **kw: MagicMock(script_path=ckpt_root / "fake.sh"),
    )
    monkeypatch.setattr(sc, "_clean_subprocess_env", lambda: {})
    monkeypatch.setattr(sc, "time", MagicMock(sleep=lambda seconds: None))


def _make_reserved(tmp_path, benches, *, max_consecutive=2):
    from leap_finetune.evaluation.async_eval_config import AsyncEvalConfig
    from leap_finetune.evaluation.reserved_callback import ReservedEvalCallback

    return ReservedEvalCallback(
        benchmarks=benches,
        cfg=AsyncEvalConfig.from_dict(
            {"mode": "reserved", "failure": {"max_consecutive": max_consecutive}}
        ),
        server_url="http://localhost:8100",
        output_dir=str(tmp_path),
        eval_gpu_ids="0",
    )


def _bench(*, name, raises=None, samples=None, metrics=None, count=1):
    from leap_finetune.evaluation.base import BenchmarkResult

    bench = MagicMock(name=name)
    bench.name = name
    bench.get_samples.return_value = samples if samples is not None else [{"x": 1}]
    if raises is not None:
        bench.evaluate_with_backend.side_effect = raises
    else:
        bench.evaluate_with_backend.return_value = BenchmarkResult(
            metrics=metrics or {"score": 1.0},
            count=count,
        )
    return bench


def _patch_reserved_server(monkeypatch, cb):
    monkeypatch.setattr(cb, "_respawn_server", lambda ckpt: None)

    import leap_finetune.distribution.vllm_server as vllm_server

    monkeypatch.setattr(vllm_server, "wait_for_vllm_health", lambda *a, **kw: None)


def test_standalone_eval_config_materializes_relative_paths(tmp_path):
    import yaml

    from leap_finetune.config import materialize_eval_config, parse_eval_config

    bench_path = tmp_path / "bench.jsonl"
    bench_path.write_text(
        '{"messages":[{"role":"user","content":"Q?"},'
        '{"role":"assistant","content":"A"}]}\n'
    )
    cfg_path = tmp_path / "eval.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "model_name": "LFM2-1.2B",
                "modality": "text",
                "evals": {
                    "benchmarks": [
                        {
                            "name": "toy",
                            "path": "./bench.jsonl",
                            "metric": "short_answer",
                        }
                    ]
                },
                "backend": {"type": "hf"},
            }
        )
    )

    materialized = materialize_eval_config(parse_eval_config(str(cfg_path)))

    assert materialized.evals.benchmarks[0].path == str(bench_path.resolve())


def test_standalone_vllm_eval_resolves_bare_model_names(monkeypatch):
    from transformers import AutoTokenizer

    from leap_finetune.evaluation import runner

    calls = {}

    def fake_from_pretrained(model_ref, **kwargs):
        calls["processor_model_ref"] = model_ref
        calls["processor_kwargs"] = kwargs
        return MagicMock()

    class FakeVLLMBackend:
        def __init__(self, model_path, **kwargs):
            calls["vllm_model_path"] = model_path
            calls["vllm_kwargs"] = kwargs

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(
        "leap_finetune.evaluation.backend.VLLMInProcessBackend",
        FakeVLLMBackend,
    )

    runner.load_eval_processor("LFM2-1.2B", modality="text")
    runner.create_vllm_backend("LFM2-1.2B", {"tensor_parallel_size": 1})

    assert calls["processor_model_ref"] == "LiquidAI/LFM2-1.2B"
    assert calls["processor_kwargs"]["trust_remote_code"] is True
    assert calls["vllm_model_path"] == "LiquidAI/LFM2-1.2B"


def test_evaluate_with_backend_contracts():
    from leap_finetune.evaluation.llm_benchmarks import (
        LLMGenerationBenchmark,
        LLMLogprobBenchmark,
    )

    generation = LLMGenerationBenchmark(
        name="gen",
        path="UNUSED",
        tokenizer=None,
        metric="short_answer",
        max_new_tokens=8,
    )
    gen_result = generation.evaluate_with_backend(
        _FakeBackend(generations=["the answer is 4", "no idea"]),
        [
            {
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                ],
            },
            {
                "messages": [
                    {"role": "user", "content": "What is 3+3?"},
                    {"role": "assistant", "content": "6"},
                ],
            },
        ],
    )
    assert gen_result.count == 2
    assert gen_result.metrics["score"] == pytest.approx(1.0)

    logprob = LLMLogprobBenchmark(name="lp", path="UNUSED", tokenizer=None)
    lp_result = logprob.evaluate_with_backend(
        _FakeBackend(logprobs=[[0.1, 0.5, 0.2], [0.1, 0.2]]),
        [
            {
                "messages": [{"role": "user", "content": "Q?"}],
                "options": ["x", "y", "z"],
                "answer_id": 1,
            },
            {
                "messages": [{"role": "user", "content": "Q?"}],
                "options": ["x", "y"],
                "answer_id": 0,
            },
        ],
    )
    assert lp_result.count == 2
    assert lp_result.metrics["score"] == pytest.approx(1.0)


def test_sbatch_script_uses_active_environment_without_uv_lock(tmp_path):
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
    content = sub.script_path.read_text()

    assert sub.script_path.stat().st_mode & 0o111
    assert "python -m leap_finetune.evaluation.async_runner_main" in content
    assert "uv run" not in content
    assert "trap 'rm -f" in content


def test_sidecar_submit_retry_does_not_leak_markers(tmp_path, monkeypatch):
    import leap_finetune.evaluation.sidecar_callback as sc

    cb = _make_sidecar(tmp_path)
    _patch_submit_prereqs(monkeypatch, ckpt_root=tmp_path)

    results = iter(
        [
            _FakeCompletedProcess(1, stderr="busy"),
            _FakeCompletedProcess(0, "Submitted batch job 999\n"),
        ]
    )
    monkeypatch.setattr(sc.subprocess, "run", lambda *a, **kw: next(results))
    assert (
        cb._submit(_make_model_mock(), MagicMock(global_step=7), MagicMock()) == "999"
    )
    assert (tmp_path / "_async_eval" / ".in_flight.step_7").read_text() == "999:7"

    fail_dir = tmp_path / "fail"
    failing_cb = _make_sidecar(fail_dir)
    _patch_submit_prereqs(monkeypatch, ckpt_root=fail_dir)
    monkeypatch.setattr(
        sc.subprocess,
        "run",
        lambda *a, **kw: _FakeCompletedProcess(1, stderr="permanent failure"),
    )

    with pytest.raises(RuntimeError, match="after 3 attempt"):
        failing_cb._submit(_make_model_mock(), MagicMock(global_step=11), MagicMock())
    assert not list((fail_dir / "_async_eval").glob(".in_flight.step_*"))


def test_sidecar_orphan_markers_disable_callback(tmp_path, monkeypatch):
    import leap_finetune.evaluation.sidecar_callback as sc

    cb = _make_sidecar(
        tmp_path,
        failure_overrides={"max_consecutive": 2},
        with_benchmarks=True,
    )
    eval_dir = cb._eval_dir
    for step, jobid in [(100, 111), (200, 222), (300, 333)]:
        (eval_dir / f".in_flight.step_{step}").write_text(f"{jobid}:{step}")

    monkeypatch.setattr(
        sc.subprocess,
        "run",
        lambda *a, **kw: _FakeCompletedProcess(0, "FAILED\n"),
    )

    cb._sweep_stale_markers()

    assert cb._consecutive_failures == 3
    assert cb._disabled is True
    assert not list(eval_dir.glob(".in_flight.step_*"))


def test_sidecar_queue_uses_per_step_markers(tmp_path):
    from leap_finetune.evaluation.async_eval_config import AsyncEvalConfig
    from leap_finetune.evaluation.sidecar_callback import (
        _MARKER_GLOB,
        SidecarEvalCallback,
    )

    cb = SidecarEvalCallback(
        benchmarks=[MagicMock(name="bench1")],
        cfg=AsyncEvalConfig.from_dict({"mode": "sidecar", "on_overlap": "queue"}),
        benchmark_configs={"benchmarks": []},
        output_dir=str(tmp_path),
        wandb_run_id=None,
    )
    eval_dir = cb._eval_dir
    (eval_dir / ".in_flight.step_1000").write_text("111:1000")
    (eval_dir / ".in_flight.step_2000").write_text("222:2000")

    (eval_dir / ".in_flight.step_1000").unlink()

    in_flight = list(eval_dir.glob(_MARKER_GLOB))
    assert len(in_flight) == 1
    assert in_flight[0].name == ".in_flight.step_2000"


def test_reserved_failure_accounting_and_in_flight_skip(tmp_path, monkeypatch):
    from leap_finetune.evaluation.reserved_callback import _EvalResult

    cb = _make_reserved(tmp_path, [MagicMock(name="bench1")], max_consecutive=2)
    for step in range(3):
        cb._account_result(_EvalResult(step=step, metrics={}, ok=True))
    assert cb._consecutive_failures == 0

    cb._account_result(_EvalResult(step=10, metrics={}, ok=False))
    cb._account_result(_EvalResult(step=11, metrics={}, ok=False))
    assert cb._disabled is True

    skip_cb = _make_reserved(tmp_path / "skip", [MagicMock(name="bench1")])
    fake_ckpt = tmp_path / "ckpt"
    fake_ckpt.mkdir()
    monkeypatch.setattr(skip_cb, "_ensure_thread", lambda: None)
    monkeypatch.setattr(skip_cb, "_save_checkpoint", lambda model, state: fake_ckpt)
    monkeypatch.setattr(
        "leap_finetune.evaluation.reserved_callback.is_rank_zero",
        lambda: True,
    )

    skip_cb.on_evaluate(
        MagicMock(),
        MagicMock(global_step=1),
        MagicMock(),
        model=MagicMock(),
    )
    skip_cb._input_q.get_nowait()
    skip_cb.on_evaluate(
        MagicMock(),
        MagicMock(global_step=2),
        MagicMock(),
        model=MagicMock(),
    )

    assert skip_cb._input_q.empty()


def test_reserved_cycle_classifies_real_failures(tmp_path, monkeypatch):
    from leap_finetune.evaluation.reserved_callback import _EvalRequest

    all_fail = _make_reserved(
        tmp_path / "all_fail",
        [
            _bench(name="b1", raises=RuntimeError("boom")),
            _bench(name="b2", raises=RuntimeError("boom")),
        ],
    )
    _patch_reserved_server(monkeypatch, all_fail)
    results, ok = all_fail._run_one_cycle(
        MagicMock(),
        _EvalRequest(step=1, ckpt_path=tmp_path / "ckpt"),
    )
    assert results == {}
    assert ok is False

    unsupported = _make_reserved(
        tmp_path / "unsupported",
        [_bench(name="b1", raises=NotImplementedError("no logprobs"))],
    )
    _patch_reserved_server(monkeypatch, unsupported)
    results, ok = unsupported._run_one_cycle(
        MagicMock(),
        _EvalRequest(step=2, ckpt_path=tmp_path / "ckpt"),
    )
    assert results == {}
    assert ok is True

    partial = _make_reserved(
        tmp_path / "partial",
        [
            _bench(name="b1", raises=RuntimeError("boom")),
            _bench(name="b2", metrics={"score": 2.0}, count=2),
        ],
    )
    _patch_reserved_server(monkeypatch, partial)
    results, ok = partial._run_one_cycle(
        MagicMock(),
        _EvalRequest(step=3, ckpt_path=tmp_path / "ckpt"),
    )
    assert results["benchmark/b2/score"] == pytest.approx(1.0)
    assert ok is True


def test_async_runner_pins_wandb_axes_before_log(monkeypatch):
    import sys

    from leap_finetune.evaluation import async_runner_main as arm

    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    arm._log_to_wandb(
        MagicMock(wandb_run_id="run-xyz", wandb_project=None, trigger_step=42),
        {
            "benchmark/refcoco/score": 0.7,
            "benchmark/gsm8k/score": 0.5,
            "train/loss": 0.1,
        },
    )

    first_log = next(i for i, call in enumerate(fake_wandb.calls) if call[0] == "log")
    defined_keys = [
        call[1][0] for call in fake_wandb.calls if call[0] == "define_metric"
    ]
    define_indices = [
        i for i, call in enumerate(fake_wandb.calls) if call[0] == "define_metric"
    ]

    assert max(define_indices) < first_log
    assert "benchmark/step" in defined_keys
    assert "benchmark/refcoco/score" in defined_keys
    assert "benchmark/gsm8k/score" in defined_keys
    assert "benchmark/*" in defined_keys
    assert "train/loss" not in defined_keys
