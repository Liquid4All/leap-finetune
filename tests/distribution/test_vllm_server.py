import pytest

from leap_finetune.distribution.vllm_server import (
    plan_gpu_split,
    resolve_vllm_rollout_plan,
)

pytestmark = pytest.mark.distribution

# === vLLM Server Resource Planning ===


def test_colocate_uses_all_gpus_without_local_server():
    plan = resolve_vllm_rollout_plan(4, {}, vllm_mode="colocate", is_multi_node=False)

    assert plan.server_gpu_ids == []
    assert plan.training_gpu_ids == [0, 1, 2, 3]
    assert plan.num_training_workers == 4
    assert not plan.launches_local_server


def test_server_split_keeps_server_out_of_ray_and_preserves_cuda_mapping(
    monkeypatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")

    plan = resolve_vllm_rollout_plan(
        4,
        {"server_gpus": 1},
        vllm_mode="server",
        is_multi_node=False,
    )

    assert plan.server_gpu_ids == [0]
    assert plan.training_gpu_ids == [1, 2, 3]
    assert plan.server_cuda_visible_devices == "4"
    assert plan.training_cuda_visible_devices == "5,6,7"
    assert plan.resources_per_worker == {"GPU": 1.0}
    assert plan.uses_custom_training_visibility


def test_server_split_can_reserve_local_judge_between_server_and_training():
    plan = resolve_vllm_rollout_plan(
        4,
        {"server_gpus": 1, "judge_gpus": 1},
        vllm_mode="server",
        is_multi_node=False,
        reserve_judge=True,
    )

    assert plan.server_gpu_ids == [0]
    assert plan.judge_gpu_ids == [1]
    assert plan.training_gpu_ids == [2, 3]
    assert plan.server_cuda_visible_devices == "0"
    assert plan.judge_cuda_visible_devices == "1"
    assert plan.training_cuda_visible_devices == "2,3"


def test_training_gpu_count_can_infer_local_server_pool():
    plan = resolve_vllm_rollout_plan(
        4,
        {"training_gpus": 2},
        vllm_mode="server",
        is_multi_node=False,
    )

    assert plan.server_gpu_ids == [0, 1]
    assert plan.training_gpu_ids == [2, 3]
    assert plan.num_training_workers == 2
    assert plan.launches_local_server


def test_external_multi_node_server_does_not_partition_local_gpus():
    plan = resolve_vllm_rollout_plan(
        4,
        {"tensor_parallel_size": 1},
        vllm_mode="server",
        is_multi_node=True,
    )

    assert not plan.launches_local_server
    assert plan.training_gpu_ids == [0, 1, 2, 3]
    assert not plan.uses_custom_training_visibility


@pytest.mark.parametrize(
    ("available", "cfg", "multi_node", "error_type", "match"),
    [
        (4, {"server_gpus": 2, "training_gpus": 3}, False, ValueError, "requests"),
        (4, {"training_gpus": 4}, False, ValueError, "local vLLM server"),
        (2, {"server_gpus": 2}, False, ValueError, "training_gpus"),
        (4, {"dedicated_gpus": 1}, True, NotImplementedError, "single-node"),
    ],
)
def test_invalid_local_partitions_raise(
    available,
    cfg,
    multi_node,
    error_type,
    match,
):
    with pytest.raises(error_type, match=match):
        resolve_vllm_rollout_plan(
            available,
            cfg,
            vllm_mode="server",
            is_multi_node=multi_node,
        )


def test_legacy_plan_gpu_split_uses_new_planner():
    assert plan_gpu_split(4, {"dedicated_gpus": 2}) == ([0, 1], [2, 3])
