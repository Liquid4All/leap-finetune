import json
import pathlib

import pytest

from leap_finetune import run_config
from leap_finetune.cli.main import main

from conftest import BASE_SFT_DATASET, write_config

pytestmark = pytest.mark.configs


def _state_file() -> pathlib.Path:
    from leap_finetune.state import get_state_dir

    return get_state_dir() / "state.json"


def _load_state() -> dict:
    return json.loads(_state_file().read_text())


def test_eval_run_writes_state_and_memory_stays_separate(
    tmp_path,
    monkeypatch,
    capsys,
):
    def fake_run_eval_config(config, *, output_path=None):
        return {"benchmark/tiny_qa/score": 1.0}

    monkeypatch.setattr(
        "leap_finetune.cli.main.check_and_handle_slurm",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        "leap_finetune.distribution.backends.kuberay.check_and_handle_kuberay",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        "leap_finetune.distribution.backends.modal.check_and_handle_modal",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        "leap_finetune.cli.main._assert_local_cuda_available",
        lambda: pytest.fail("eval-only run_config should not require CUDA"),
    )
    monkeypatch.setattr(
        "leap_finetune.evaluation.runner.run_eval_config",
        fake_run_eval_config,
    )

    cfg_path = write_config(
        {
            "model_name": "LFM2-1.2B",
            "evals": {
                "benchmarks": [
                    {
                        "name": "tiny_qa",
                        "path": "/tmp/tiny_qa.jsonl",
                        "metric": "short_answer",
                    }
                ]
            },
        },
        tmp_path,
    )

    result = run_config(cfg_path)
    state = _load_state()
    run = state["runs"][0]

    assert result == {"benchmark/tiny_qa/score": 1.0}
    assert run["status"] == "completed"
    assert run["kind"] == "eval"
    assert run["metrics"] == {"benchmark/tiny_qa/score": 1.0}
    assert pathlib.Path(run["config"]) == pathlib.Path(cfg_path)

    monkeypatch.setattr("sys.argv", ["leap-finetune", "runs", "list"])
    main()
    assert run["id"] in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["leap-finetune", "runs", "report"])
    main()
    report = capsys.readouterr().out
    assert run["id"] in report
    assert "last_eval" in report

    monkeypatch.setattr(
        "sys.argv",
        [
            "leap-finetune",
            "memory",
            "add",
            "Tiny QA passed; next run should broaden the benchmark.",
            "--ref",
            run["id"],
        ],
    )
    main()

    memory = pathlib.Path(_state_file()).with_name("memory.md").read_text()
    assert f"Refs: `{run['id']}`" in memory
    assert "Tiny QA passed" in memory
    assert "benchmark/tiny_qa/score" not in memory


def test_remote_submission_writes_backend_state(tmp_path, monkeypatch):
    calls = {}

    monkeypatch.setattr(
        "leap_finetune.cli.main.check_and_handle_slurm",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        "leap_finetune.distribution.backends.kuberay.check_and_handle_kuberay",
        lambda *args, **kwargs: False,
    )

    def fake_modal(config_path_arg=None, *, config_dict=None):
        calls["config_path_arg"] = config_path_arg
        calls["config_dict"] = config_dict
        return {"status": "submitted", "app_id": "ap-test"}

    monkeypatch.setattr(
        "leap_finetune.distribution.backends.modal.check_and_handle_modal",
        fake_modal,
    )
    monkeypatch.setattr(
        "leap_finetune.cli.main._assert_local_cuda_available",
        lambda: pytest.fail("remote config should not require CUDA"),
    )

    cfg_path = write_config(
        {
            "project_name": "modal_run",
            "model_name": "LFM2-1.2B",
            "training_type": "sft",
            "dataset": BASE_SFT_DATASET,
            "training_config": {"num_train_epochs": 1},
            "modal": {"gpu": "H100"},
        },
        tmp_path,
    )

    run_config(cfg_path)
    run = _load_state()["runs"][0]

    assert calls["config_path_arg"] == str(pathlib.Path(cfg_path).resolve())
    assert calls["config_dict"]["modal"]["gpu"] == "H100"
    assert run["status"] == "submitted"
    assert run["kind"] == "train"
    assert run["backend"] == "modal"
    assert run["backend_id"] == "ap-test"


def test_state_v2_defaults_and_progress_history(tmp_path):
    from leap_finetune.state import (
        RunTracker,
        list_runs,
        load_run,
        record_checkpoint,
        record_eval_result,
        render_runs_report,
        update_run_progress,
    )

    legacy_state = {
        "version": 1,
        "updated_at": "2026-01-01T00:00:00+00:00",
        "runs": [
            {
                "id": "legacy-run",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "status": "running",
                "kind": "train",
                "metrics": {},
            }
        ],
    }
    _state_file().parent.mkdir(parents=True, exist_ok=True)
    _state_file().write_text(json.dumps(legacy_state))

    legacy = list_runs()[0]
    assert legacy["phase"] == "training"
    assert legacy["progress"]["last_log"] is None
    assert legacy["history"]["logs"] == []
    assert legacy["log_refs"] == {}

    tracker = RunTracker.start(
        config_path=None,
        config_dict={"training_type": "sft", "dataset": BASE_SFT_DATASET},
        output_path=tmp_path / "out",
    )
    for step in range(250):
        update_run_progress(
            tracker.id,
            step=step,
            max_steps=300,
            log={"loss": 1.0 / (step + 1), "nested": {"ignored": True}},
        )
    record_eval_result(
        tracker.id,
        step=249,
        metrics={
            "benchmark/tiny_qa/score": 0.75,
            "benchmark/tool_call/accuracy": 0.5,
        },
        source="benchmark",
    )
    record_checkpoint(tracker.id, path=tmp_path / "out" / "checkpoint-249", step=249)

    run = load_run(tracker.id)
    assert len(run["history"]["logs"]) == 200
    assert run["progress"]["step"] == 249
    assert run["progress"]["max_steps"] == 300
    assert run["progress"]["last_log"]["metrics"]["loss"] == pytest.approx(1 / 250)
    assert "nested" not in run["progress"]["last_log"]["metrics"]
    assert run["progress"]["last_eval"]["metrics"]["benchmark/tiny_qa/score"] == 0.75
    assert run["progress"]["last_checkpoint"]["path"].endswith("checkpoint-249")
    assert "benchmark/tool_call/accuracy" in run["history"]["evals"][-1]["metrics"]
    assert tracker.id in render_runs_report([run])


def test_lft_state_callback_records_trainer_events(tmp_path):
    from transformers import TrainingArguments
    from transformers.trainer_callback import TrainerControl, TrainerState

    from leap_finetune.state import RunTracker, load_run
    from leap_finetune.state.callback import LFTStateCallback

    tracker = RunTracker.start(
        config_path=None,
        config_dict={"training_type": "sft", "dataset": BASE_SFT_DATASET},
        output_path=tmp_path / "out",
    )
    callback = LFTStateCallback(run_id=tracker.id)
    args = TrainingArguments(
        output_dir=str(tmp_path / "out"),
        max_steps=10,
        report_to=[],
    )
    state = TrainerState(global_step=3, epoch=0.5, max_steps=10)
    control = TrainerControl()

    callback.on_train_begin(args, state, control)
    callback.on_log(args, state, control, logs={"loss": 0.4, "tokens": 128})
    callback.on_evaluate(args, state, control, metrics={"eval_loss": 0.3})
    callback.on_save(args, state, control)
    callback.on_train_end(args, state, control)

    run = load_run(tracker.id)
    assert run["status"] == "completed"
    assert run["phase"] == "completed"
    assert run["progress"]["max_steps"] == 10
    assert run["history"]["logs"][-1]["metrics"]["loss"] == 0.4
    assert run["history"]["evals"][-1]["metrics"]["eval_loss"] == 0.3
    assert run["history"]["checkpoints"][-1]["step"] == 3
