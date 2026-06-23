from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
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


def _make_sidecar(
    tmp_path,
    *,
    failure_overrides=None,
    with_benchmarks=False,
    cfg_overrides=None,
):
    from leap_finetune.evaluation.async_eval_config import AsyncEvalConfig
    from leap_finetune.evaluation.sidecar_callback import SidecarEvalCallback

    failure = {
        "max_consecutive": 99,
        "max_submit_attempts": 3,
        "submit_retry_backoff": 1.0,
    }
    if failure_overrides:
        failure.update(failure_overrides)
    raw_cfg = {"mode": "sidecar", "failure": failure}
    if cfg_overrides:
        raw_cfg.update(cfg_overrides)

    return SidecarEvalCallback(
        benchmarks=[MagicMock(name="bench1")] if with_benchmarks else [],
        cfg=AsyncEvalConfig.from_dict(raw_cfg),
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
        lambda **kw: SimpleNamespace(
            script_path=ckpt_root / "fake.sh",
            log_out=ckpt_root / "fake.out",
            log_err=ckpt_root / "fake.err",
        ),
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


def _start_state_run(tmp_path, monkeypatch, run_id: str):
    from leap_finetune.state import RunTracker

    state_dir = tmp_path / ".lft"
    monkeypatch.setenv("LFT_STATE_DIR", str(state_dir))
    monkeypatch.setenv("LFT_RUN_ID", run_id)
    return RunTracker.start(
        config_path=None,
        config_dict={"project_name": run_id, "training_type": "sft"},
        output_path=tmp_path,
        state_dir=state_dir,
    )


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


def test_sidecar_submit_retries_and_records_submission(tmp_path, monkeypatch):
    import leap_finetune.evaluation.sidecar_callback as sc

    tracker = _start_state_run(tmp_path, monkeypatch, "sidecar-submit")
    cb = _make_sidecar(tmp_path, with_benchmarks=True)
    _patch_submit_prereqs(monkeypatch, ckpt_root=tmp_path)
    monkeypatch.setattr(
        "leap_finetune.evaluation.sidecar_callback.is_rank_zero",
        lambda: True,
    )

    attempts = []
    results = iter(
        [
            _FakeCompletedProcess(1, stderr="busy"),
            _FakeCompletedProcess(0, "Submitted batch job 999\n"),
        ]
    )
    monkeypatch.setattr(
        sc.subprocess,
        "run",
        lambda *a, **kw: attempts.append(a) or next(results),
    )

    cb.on_step_end(
        MagicMock(),
        SimpleNamespace(global_step=7),
        SimpleNamespace(should_evaluate=True),
        model=_make_model_mock(),
    )

    run = tracker.load()
    last_eval = run["progress"]["last_eval"]
    assert len(attempts) == 2
    assert last_eval["status"] == "submitted"
    assert last_eval["source"] == "async_sidecar_submit"
    assert last_eval["step"] == 7
    assert last_eval["backend"] == "slurm"
    assert last_eval["backend_id"] == "999"
    assert "async_eval_step_7_stdout" in run["log_refs"]


def test_sidecar_repeated_submission_failures_stop_future_submissions(
    tmp_path, monkeypatch
):
    import leap_finetune.evaluation.sidecar_callback as sc

    tracker = _start_state_run(tmp_path, monkeypatch, "sidecar-failure")
    cb = _make_sidecar(
        tmp_path,
        failure_overrides={"max_consecutive": 2, "max_submit_attempts": 1},
        with_benchmarks=True,
    )
    _patch_submit_prereqs(monkeypatch, ckpt_root=tmp_path)
    monkeypatch.setattr(
        "leap_finetune.evaluation.sidecar_callback.is_rank_zero",
        lambda: True,
    )

    attempts = []
    monkeypatch.setattr(
        sc.subprocess,
        "run",
        lambda *a, **kw: attempts.append(a)
        or _FakeCompletedProcess(1, stderr="permanent failure"),
    )

    for step in (1, 2, 3):
        cb.on_step_end(
            MagicMock(),
            SimpleNamespace(global_step=step),
            SimpleNamespace(should_evaluate=True),
            model=_make_model_mock(),
        )

    failed_records = [
        record
        for record in tracker.load()["history"]["evals"]
        if record["status"] == "failed"
    ]
    assert len(attempts) == 2
    assert [record["step"] for record in failed_records] == [1, 2]
    assert failed_records[-1]["source"] == "async_sidecar_submit"


def test_reserved_skips_new_submission_while_eval_is_in_flight(tmp_path, monkeypatch):
    tracker = _start_state_run(tmp_path, monkeypatch, "reserved-overlap")
    cb = _make_reserved(tmp_path, [MagicMock(name="bench1")])
    fake_ckpt = tmp_path / "ckpt"
    fake_ckpt.mkdir()
    saved_steps = []
    monkeypatch.setattr(cb, "_ensure_thread", lambda: None)
    monkeypatch.setattr(
        cb,
        "_save_checkpoint",
        lambda model, state: saved_steps.append(state.global_step) or fake_ckpt,
    )
    monkeypatch.setattr(
        "leap_finetune.evaluation.reserved_callback.is_rank_zero",
        lambda: True,
    )

    cb.on_evaluate(
        MagicMock(),
        SimpleNamespace(global_step=1),
        MagicMock(),
        model=MagicMock(),
    )
    cb.on_evaluate(
        MagicMock(),
        SimpleNamespace(global_step=2),
        MagicMock(),
        model=MagicMock(),
    )

    evals = tracker.load()["history"]["evals"]
    assert saved_steps == [1]
    assert len(evals) == 1
    assert evals[0]["status"] == "submitted"
    assert evals[0]["source"] == "async_reserved"


def test_reserved_result_drain_records_completed_and_failed_state(
    tmp_path, monkeypatch
):
    tracker = _start_state_run(tmp_path, monkeypatch, "reserved-results")
    cb = _make_reserved(tmp_path, [MagicMock(name="bench1")], max_consecutive=1)
    monkeypatch.setattr(
        "leap_finetune.evaluation.reserved_callback.is_rank_zero",
        lambda: True,
    )

    cb._output_q.put(
        SimpleNamespace(
            step=10,
            metrics={"benchmark/b1/score": 0.5},
            ok=True,
        )
    )
    cb.on_log(MagicMock(), SimpleNamespace(global_step=10), MagicMock())
    completed = tracker.load()["progress"]["last_eval"]

    cb._output_q.put(
        SimpleNamespace(
            step=11,
            metrics={},
            ok=False,
        )
    )
    cb.on_log(MagicMock(), SimpleNamespace(global_step=11), MagicMock())
    failed = tracker.load()["progress"]["last_eval"]

    assert completed["status"] == "completed"
    assert completed["metrics"]["benchmark/b1/score"] == pytest.approx(0.5)
    assert failed["status"] == "failed"
    assert failed["source"] == "async_reserved"


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
