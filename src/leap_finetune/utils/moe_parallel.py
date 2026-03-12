import logging

import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)


# === EP Process Group Setup (Step 8) ===


def create_ep_process_groups(ep_size: int, num_experts: int) -> dict:
    """Create Expert Parallelism process groups for AlltoAll communication.

    Partitions world into groups of `ep_size` ranks. Each group forms an EP group
    where experts are distributed via AlltoAll.

    Args:
        ep_size: number of ranks in each EP group
        num_experts: total number of experts in the model

    Returns:
        Dict with ep_group, ep_rank, ep_size, n_local_experts, local_expert_indices
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size % ep_size != 0:
        raise ValueError(
            f"World size ({world_size}) must be divisible by ep_size ({ep_size})"
        )
    if num_experts % ep_size != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})"
        )

    n_local_experts = num_experts // ep_size
    my_ep_group = None
    my_ep_rank = 0

    for i in range(0, world_size, ep_size):
        ranks = list(range(i, i + ep_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            my_ep_group = group
            my_ep_rank = ranks.index(rank)

    local_expert_indices = list(
        range(
            my_ep_rank * n_local_experts,
            (my_ep_rank + 1) * n_local_experts,
        )
    )

    logger.info(
        f"EP rank {my_ep_rank}/{ep_size}: local experts {local_expert_indices}"
    )

    return {
        "ep_group": my_ep_group,
        "ep_rank": my_ep_rank,
        "ep_size": ep_size,
        "n_local_experts": n_local_experts,
        "local_expert_indices": local_expert_indices,
        "num_experts": num_experts,
    }


# === Expert Weight Sharding (Step 9) ===


def shard_experts(model: nn.Module, ep_config: dict) -> None:
    """Remove non-local expert weights to free memory.

    After sharding, each GPU keeps only its assigned experts. The ModuleList
    is replaced with only local experts, and metadata is stored for routing.

    Args:
        model: the full model with all experts loaded
        ep_config: dict from create_ep_process_groups()
    """
    local_indices = ep_config["local_expert_indices"]
    n_local = ep_config["n_local_experts"]
    sharded = 0

    for module in model.modules():
        if type(module).__name__ != "Lfm2MoeExperts":
            continue

        # Keep only local experts
        local_experts = nn.ModuleList([module[i] for i in local_indices])
        module.clear()
        module.extend(local_experts)

        # Store index mapping for routing
        module._local_expert_offset = local_indices[0]
        module._n_local_experts = n_local
        sharded += 1

    logger.info(
        f"Sharded {sharded} expert modules: keeping {n_local} of {ep_config['num_experts']} experts"
    )


# === EP-Aware FSDP Config (Step 11) ===

MOE_FSDP_CONFIG_EP = {
    "fsdp": ["hybrid_shard", "auto_wrap"],
    "fsdp_config": {
        "transformer_layer_cls_to_wrap": "Lfm2MoeDecoderLayer",
        "backward_prefetch": "backward_pre",
        "sync_module_states": True,
        "use_orig_params": True,
        "activation_checkpointing": True,
    },
}
