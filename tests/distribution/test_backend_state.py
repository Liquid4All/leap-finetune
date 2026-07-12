import pytest

pytestmark = pytest.mark.distribution


def test_slurm_log_refs_expand_job_and_name(tmp_path):
    from leap_finetune.distribution.backends.slurm import (
        _resolve_slurm_settings,
        _slurm_log_refs,
    )

    settings = _resolve_slurm_settings(
        {
            "project_name": "state-test",
            "training_type": "sft",
            "dataset": {"path": "dummy", "type": "sft"},
            "slurm": {
                "output": "logs/OUT_%x.%j",
                "error": "logs/ERR_%x.%j",
            },
        }
    )
    refs = _slurm_log_refs(settings, job_id="1234", submit_cwd=tmp_path)

    assert refs["stdout"] == str(tmp_path / "logs" / "OUT_state-test.1234")
    assert refs["stderr"] == str(tmp_path / "logs" / "ERR_state-test.1234")
    assert refs["slurm_stdout_template"] == "logs/OUT_%x.%j"


def test_kuberay_manifest_injects_state_env(monkeypatch):
    from leap_finetune.distribution.backends.kuberay import _generate_rayjob_manifest

    monkeypatch.setenv("LFT_RUN_ID", "run-123")
    manifest = _generate_rayjob_manifest(
        "job-1",
        "job-1-config",
        {
            "image": "example/leap:latest",
            "worker_replicas": 1,
            "gpus_per_worker": 1,
            "state_dir": "/outputs/custom-state",
        },
        "/outputs",
    )

    head_env = {
        item["name"]: item["value"]
        for item in manifest["spec"]["rayClusterSpec"]["headGroupSpec"]["template"][
            "spec"
        ]["containers"][0]["env"]
    }
    worker_env = {
        item["name"]: item["value"]
        for item in manifest["spec"]["rayClusterSpec"]["workerGroupSpecs"][0][
            "template"
        ]["spec"]["containers"][0]["env"]
    }

    assert head_env["LFT_RUN_ID"] == "run-123"
    assert head_env["LFT_STATE_DIR"] == "/outputs/custom-state"
    assert worker_env["LFT_RUN_ID"] == "run-123"
    assert worker_env["LFT_STATE_DIR"] == "/outputs/custom-state"
