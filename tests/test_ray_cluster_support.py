import pathlib
import shlex
import subprocess
import textwrap

from leap_finetune.trainer import _resolve_num_workers
from leap_finetune.utils.logging_utils import (
    get_requested_ray_address,
    should_connect_existing_cluster,
)
from leap_finetune.utils.slurm_generator import generate_slurm_script


def test_get_requested_ray_address_prefers_leap_env(monkeypatch):
    monkeypatch.setenv("RAY_ADDRESS", "ray-a:6379")
    monkeypatch.setenv("LEAP_RAY_ADDRESS", "ray-b:6379")
    assert get_requested_ray_address({"address": "ray-c:6379"}) == "ray-b:6379"


def test_get_requested_ray_address_uses_config(monkeypatch):
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv("LEAP_RAY_ADDRESS", raising=False)
    assert get_requested_ray_address({"address": "ray-c:6379"}) == "ray-c:6379"
    assert should_connect_existing_cluster({"address": "ray-c:6379"}) is True


def test_resolve_num_workers_prefers_env(monkeypatch):
    monkeypatch.setenv("LEAP_RAY_NUM_WORKERS", "16")
    assert (
        _resolve_num_workers(
            None,
            local_num_gpus=8,
            connected_to_existing_cluster=False,
        )
        == 16
    )


def test_resolve_num_workers_uses_ray_config(monkeypatch):
    monkeypatch.delenv("LEAP_RAY_NUM_WORKERS", raising=False)
    monkeypatch.delenv("LEAP_NUM_WORKERS", raising=False)
    assert (
        _resolve_num_workers(
            {"num_workers": 12},
            local_num_gpus=8,
            connected_to_existing_cluster=False,
        )
        == 12
    )


def test_resolve_num_workers_uses_local_gpu_count(monkeypatch):
    monkeypatch.delenv("LEAP_RAY_NUM_WORKERS", raising=False)
    monkeypatch.delenv("LEAP_NUM_WORKERS", raising=False)
    assert (
        _resolve_num_workers(
            None,
            local_num_gpus=8,
            connected_to_existing_cluster=False,
        )
        == 8
    )


def test_multinode_slurm_script_starts_ray_cluster(tmp_path):
    config_path = tmp_path / "example.yaml"
    config_path.write_text(
        """
project_name: test_multinode
model_name: LFM2-1.2B
training_type: sft
dataset:
  path: HuggingFaceTB/smoltalk
  type: sft
training_config:
  extends: DEFAULT_SFT
peft_config:
  use_peft: false
slurm:
  nodes: 2
  ntasks_per_node: 1
  gpus_per_task: 8
"""
    )

    script_path = generate_slurm_script(
        config_path,
        {
            "project_name": "test_multinode",
            "slurm": {
                "nodes": 2,
                "ntasks_per_node": 1,
                "gpus_per_task": 8,
            },
        },
        tmp_path,
    )
    script = script_path.read_text()
    assert "source job_configs/slurms/utils/slurm_ray.sh" in script
    assert "export RAY_ADDRESS" in script
    assert "ray_slurm_start_cluster_bg" in script


def test_ray_slurm_stop_cluster_does_not_create_new_srun_steps():
    helper_path = pathlib.Path("job_configs/slurms/utils/slurm_ray.sh").resolve()
    script = textwrap.dedent(
        f"""
        set -euo pipefail
        source {shlex.quote(str(helper_path))}

        srun() {{
          echo "unexpected srun call: $*" >&2
          exit 64
        }}

        export RAY_SLURM_STOP_TIMEOUT=2
        sleep 60 &
        ray_pid=$!
        RAY_SLURM_PIDS=("${{ray_pid}}")
        RAY_SLURM_NODES=(node0 node1)
        RAY_HEAD_NODE=node0

        ray_slurm_stop_cluster

        if kill -0 "${{ray_pid}}" 2>/dev/null; then
          echo "ray srun wrapper was not stopped" >&2
          exit 1
        fi
        """
    )

    result = subprocess.run(
        ["bash", "-lc", script],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
