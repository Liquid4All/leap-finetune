import pytest

from leap_finetune.rl.vllm_server import (
    plan_gpu_split,
    resolve_vllm_rollout_plan,
)

pytestmark = pytest.mark.configs


# === vLLM server resource planning ===


class TestVLLMRolloutResourcePlan:
    def test_colocate_uses_all_gpus_without_local_server(self):
        plan = resolve_vllm_rollout_plan(
            4, {}, vllm_mode="colocate", is_multi_node=False
        )
        assert plan.server_gpu_ids == []
        assert plan.training_gpu_ids == [0, 1, 2, 3]
        assert plan.num_training_workers == 4
        assert not plan.launches_local_server

    def test_colocate_ignores_server_split_block(self):
        plan = resolve_vllm_rollout_plan(
            4,
            {"server_gpus": 1, "training_gpus": 2},
            vllm_mode="colocate",
            is_multi_node=True,
        )
        assert plan.server_gpu_ids == []
        assert plan.training_gpu_ids == [0, 1, 2, 3]
        assert plan.num_training_workers == 4

    def test_colocate_can_reserve_local_judge_gpu(self):
        plan = resolve_vllm_rollout_plan(
            4,
            {},
            vllm_mode="colocate",
            is_multi_node=False,
            reserve_judge=True,
        )
        assert plan.server_gpu_ids == []
        assert plan.judge_gpu_ids == [0]
        assert plan.training_gpu_ids == [1, 2, 3]
        assert plan.judge_cuda_visible_devices == "0"
        assert plan.training_cuda_visible_devices == "1,2,3"
        assert plan.launches_local_judge

    def test_server_gpu_count_keeps_server_out_of_ray(self):
        plan = resolve_vllm_rollout_plan(
            4,
            {"server_gpus": 1},
            vllm_mode="server",
            is_multi_node=False,
        )
        assert plan.server_gpu_ids == [0]
        assert plan.judge_gpu_ids == []
        assert plan.training_gpu_ids == [1, 2, 3]
        assert plan.server_cuda_visible_devices == "0"
        assert plan.training_cuda_visible_devices == "1,2,3"
        assert plan.num_training_workers == 3
        assert plan.resources_per_worker == {"GPU": 1.0}
        assert plan.uses_custom_training_visibility

    def test_existing_cuda_visible_devices_mapping_is_preserved(self, monkeypatch):
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

    def test_training_gpu_count_can_leave_extra_gpus_idle(self):
        plan = resolve_vllm_rollout_plan(
            4,
            {"server_gpus": 1, "training_gpus": 2},
            vllm_mode="server",
            is_multi_node=False,
        )
        assert plan.server_gpu_ids == [0]
        assert plan.training_gpu_ids == [1, 2]
        assert plan.num_training_workers == 2

    def test_training_gpu_count_infers_remaining_server_gpus(self):
        plan = resolve_vllm_rollout_plan(
            4,
            {"training_gpus": 2},
            vllm_mode="server",
            is_multi_node=False,
        )

        assert plan.server_gpu_ids == [0, 1]
        assert plan.training_gpu_ids == [2, 3]
        assert plan.server_cuda_visible_devices == "0,1"
        assert plan.training_cuda_visible_devices == "2,3"
        assert plan.num_training_workers == 2
        assert plan.launches_local_server

    def test_server_mode_reserves_judge_between_rollout_and_training(self):
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

    def test_training_gpu_count_infers_server_after_judge_reservation(self):
        plan = resolve_vllm_rollout_plan(
            4,
            {"training_gpus": 2},
            vllm_mode="server",
            is_multi_node=False,
            reserve_judge=True,
        )

        assert plan.server_gpu_ids == [0]
        assert plan.judge_gpu_ids == [1]
        assert plan.training_gpu_ids == [2, 3]

    def test_training_gpu_count_must_leave_a_server_gpu_when_inferred(self):
        with pytest.raises(ValueError, match="local vLLM server"):
            resolve_vllm_rollout_plan(
                4,
                {"training_gpus": 4},
                vllm_mode="server",
                is_multi_node=False,
            )

    def test_server_gpus_zero_keeps_external_server_mode_explicit(self):
        plan = resolve_vllm_rollout_plan(
            4,
            {"server_gpus": 0, "training_gpus": 2},
            vllm_mode="server",
            is_multi_node=False,
        )

        assert not plan.launches_local_server
        assert plan.server_gpu_ids == []
        assert plan.training_gpu_ids == [0, 1]
        assert plan.training_cuda_visible_devices == "0,1"
        assert plan.num_training_workers == 2

    def test_judge_gpus_are_ignored_without_local_judge_reward(self):
        plan = resolve_vllm_rollout_plan(
            4,
            {"judge_gpus": 1},
            vllm_mode="colocate",
            is_multi_node=False,
            reserve_judge=False,
        )

        assert not plan.launches_local_judge
        assert plan.training_gpu_ids == [0, 1, 2, 3]

    def test_dedicated_gpus_legacy_alias_still_works(self):
        plan = resolve_vllm_rollout_plan(
            4,
            {"dedicated_gpus": 2},
            vllm_mode="server",
            is_multi_node=False,
        )
        assert plan.server_gpu_ids == [0, 1]
        assert plan.training_gpu_ids == [2, 3]
        assert plan.num_training_workers == 2

    def test_too_many_requested_gpus_raises(self):
        with pytest.raises(ValueError, match="requests"):
            resolve_vllm_rollout_plan(
                4,
                {"server_gpus": 2, "training_gpus": 3},
                vllm_mode="server",
                is_multi_node=False,
            )

    def test_training_gpu_count_must_leave_a_training_gpu(self):
        with pytest.raises(ValueError, match="training_gpus"):
            resolve_vllm_rollout_plan(
                2,
                {"server_gpus": 2},
                vllm_mode="server",
                is_multi_node=False,
            )

    def test_dedicated_local_server_is_single_node_only(self):
        with pytest.raises(NotImplementedError, match="single-node"):
            resolve_vllm_rollout_plan(
                4,
                {"dedicated_gpus": 1},
                vllm_mode="server",
                is_multi_node=True,
            )

    def test_training_gpu_count_is_single_node_only(self):
        with pytest.raises(NotImplementedError, match="single-node"):
            resolve_vllm_rollout_plan(
                4,
                {"training_gpus": 2},
                vllm_mode="server",
                is_multi_node=True,
            )

    def test_external_multi_node_server_without_local_partitioning(self):
        plan = resolve_vllm_rollout_plan(
            4,
            {"tensor_parallel_size": 1},
            vllm_mode="server",
            is_multi_node=True,
        )

        assert not plan.launches_local_server
        assert not plan.launches_local_judge
        assert plan.training_gpu_ids == [0, 1, 2, 3]
        assert not plan.uses_custom_training_visibility

    def test_local_judge_partitioning_is_single_node_only(self):
        with pytest.raises(NotImplementedError, match="single-node"):
            resolve_vllm_rollout_plan(
                4,
                {},
                vllm_mode="colocate",
                is_multi_node=True,
                reserve_judge=True,
            )

    def test_legacy_plan_gpu_split_uses_new_planner(self):
        assert plan_gpu_split(4, {"dedicated_gpus": 2}) == ([0, 1], [2, 3])
