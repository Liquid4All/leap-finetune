import logging

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# === 2D Process Group Mesh (CP x EP) ===


def create_parallel_process_groups(
    cp_size: int = 1,
    ep_size: int = 1,
    num_experts: int | None = None,
) -> dict:
    """Create orthogonal CP and EP process groups from a 2D rank mesh.

    Ranks are laid out as a 3D grid: [dp, cp, ep] where
    world_size = dp_size * cp_size * ep_size.

    CP groups: ranks that differ only in cp dimension (same dp, same ep slot).
    EP groups: ranks that differ only in ep dimension (same dp, same cp slot).

    Args:
        cp_size: number of ranks per context-parallel group
        ep_size: number of ranks per expert-parallel group
        num_experts: total experts in MoE model (required when ep_size > 1)
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    mesh_size = cp_size * ep_size

    if world_size % mesh_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by "
            f"cp_size * ep_size ({cp_size} * {ep_size} = {mesh_size})"
        )
    if ep_size > 1 and num_experts is not None and num_experts % ep_size != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})"
        )

    dp_size = world_size // mesh_size

    # Rank coordinates in [dp, cp, ep] grid
    # rank = dp_idx * (cp_size * ep_size) + cp_idx * ep_size + ep_idx
    dp_idx = rank // mesh_size
    remainder = rank % mesh_size
    cp_idx = remainder // ep_size
    ep_idx = remainder % ep_size

    # === CP groups: vary cp_idx, fix dp_idx and ep_idx ===
    my_cp_group = None
    for d in range(dp_size):
        for e in range(ep_size):
            ranks = [d * mesh_size + c * ep_size + e for c in range(cp_size)]
            group = dist.new_group(ranks)
            if rank in ranks:
                my_cp_group = group

    # === EP groups: vary ep_idx, fix dp_idx and cp_idx ===
    my_ep_group = None
    n_local_experts = None
    local_expert_indices = None
    if ep_size > 1:
        n_local_experts = num_experts // ep_size if num_experts else None
        for d in range(dp_size):
            for c in range(cp_size):
                ranks = [d * mesh_size + c * ep_size + e for e in range(ep_size)]
                group = dist.new_group(ranks)
                if rank in ranks:
                    my_ep_group = group

        if n_local_experts is not None:
            local_expert_indices = list(
                range(ep_idx * n_local_experts, (ep_idx + 1) * n_local_experts)
            )

    result = {
        "cp_group": my_cp_group,
        "cp_rank": cp_idx,
        "cp_size": cp_size,
    }

    if ep_size > 1:
        result.update(
            {
                "ep_group": my_ep_group,
                "ep_rank": ep_idx,
                "ep_size": ep_size,
                "n_local_experts": n_local_experts,
                "local_expert_indices": local_expert_indices,
                "num_experts": num_experts,
            }
        )

    logger.info(f"Rank {rank}: dp={dp_idx} cp={cp_idx}/{cp_size} ep={ep_idx}/{ep_size}")
    return result


# === All-Gather with Gradient Support ===


class _AllGatherSeq(torch.autograd.Function):
    """All-gather along sequence dimension with reduce-scatter backward."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, group: dist.ProcessGroup, cp_size: int):
        ctx.group = group
        ctx.cp_size = cp_size
        ctx.seq_dim = 2  # [B, n_heads, S_local, head_dim]

        gathered = [torch.empty_like(input_) for _ in range(cp_size)]
        dist.all_gather(gathered, input_.contiguous(), group=group)
        return torch.cat(gathered, dim=ctx.seq_dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Split grad along seq dim, then reduce (sum) each chunk across ranks
        chunks = list(grad_output.chunk(ctx.cp_size, dim=ctx.seq_dim))
        # Each rank keeps its own chunk's gradient, reduced across the group
        cp_rank = dist.get_rank(ctx.group)
        grad_local = chunks[cp_rank].contiguous()
        dist.all_reduce(grad_local, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_local, None, None


def all_gather_seq(
    input_: torch.Tensor, group: dist.ProcessGroup, cp_size: int
) -> torch.Tensor:
    return _AllGatherSeq.apply(input_, group, cp_size)


# === CP Causal Mask ===


def build_cp_causal_mask(
    s_local: int,
    s_full: int,
    cp_rank: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build causal attention mask for a CP rank.

    Returns mask of shape [1, 1, S_local, S_full] compatible with SDPA.
    Uses -inf for masked positions (standard SDPA convention).

    Q positions are global: [cp_rank * s_local, ..., (cp_rank + 1) * s_local - 1]
    K positions are global: [0, ..., s_full - 1]
    Causal: Q at global pos i can attend to K at global pos j iff j <= i.
    """
    q_start = cp_rank * s_local
    # q_positions: [s_local], k_positions: [s_full]
    q_pos = torch.arange(q_start, q_start + s_local, device=device)
    k_pos = torch.arange(s_full, device=device)

    # mask[i, j] = True where q_pos[i] >= k_pos[j] (allowed to attend)
    causal = q_pos.unsqueeze(1) >= k_pos.unsqueeze(0)  # [S_local, S_full]

    # Convert to float mask: 0.0 for allowed, -inf for masked
    mask = torch.where(
        causal,
        torch.tensor(0.0, dtype=dtype, device=device),
        torch.tensor(float("-inf"), dtype=dtype, device=device),
    )
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S_local, S_full]


# === Attention Monkey-Patch for CP ===


def patch_attention_for_cp(
    attn: nn.Module,
    cp_group: dist.ProcessGroup,
    cp_rank: int,
    cp_size: int,
) -> None:
    """Monkey-patch an attention module for context parallelism.

    The patched forward:
    1. Computes Q, K, V from local hidden_states (S_local tokens)
    2. Applies RoPE with global position_ids (offset by cp_rank * S_local)
    3. All-gathers K, V across CP group
    4. Computes attention via SDPA with CP-aware causal mask
    5. Applies output projection
    """

    def cp_attention_forward(hidden_states: torch.Tensor, **kwargs):
        bsz, s_local, _ = hidden_states.shape
        s_full = s_local * cp_size

        # Override position_ids to use global positions for correct RoPE
        global_offset = cp_rank * s_local
        kwargs["position_ids"] = (
            torch.arange(
                global_offset,
                global_offset + s_local,
                device=hidden_states.device,
            )
            .unsqueeze(0)
            .expand(bsz, -1)
        )

        # === Q, K, V projections ===
        q = attn.q_proj(hidden_states)
        k = attn.k_proj(hidden_states)
        v = attn.v_proj(hidden_states)

        # Get head dimensions from the attention module
        num_heads = attn.config.num_attention_heads
        num_kv_heads = getattr(attn.config, "num_key_value_heads", num_heads)
        head_dim = attn.head_dim

        # Reshape: [B, S, num_heads * head_dim] → [B, num_heads, S, head_dim]
        q = q.view(bsz, s_local, num_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, s_local, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, s_local, num_kv_heads, head_dim).transpose(1, 2)

        # === Apply RoPE with global positions ===
        position_ids = kwargs["position_ids"]
        if hasattr(attn, "rotary_emb"):
            cos, sin = attn.rotary_emb(v, position_ids)
            # apply_rotary_pos_emb is shared across HF model implementations
            try:
                from transformers.models.llama.modeling_llama import (
                    apply_rotary_pos_emb,
                )
            except ImportError:
                from transformers.models.mistral.modeling_mistral import (
                    apply_rotary_pos_emb,
                )
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # === GQA: expand K, V if num_kv_heads < num_heads ===
        if num_kv_heads < num_heads:
            n_rep = num_heads // num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # === All-gather K, V across CP group ===
        # k, v: [B, num_heads, S_local, head_dim] → [B, num_heads, S_full, head_dim]
        k_full = all_gather_seq(k, cp_group, cp_size)
        v_full = all_gather_seq(v, cp_group, cp_size)

        # === SDPA with CP causal mask ===
        mask = build_cp_causal_mask(
            s_local, s_full, cp_rank, hidden_states.device, q.dtype
        )
        attn_output = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=mask)
        # attn_output: [B, num_heads, S_local, head_dim]

        # === Output projection ===
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, s_local, -1)
        attn_output = attn.o_proj(attn_output)

        # Return in the same format as the original forward
        # HF attention returns (attn_output, attn_weights, past_key_value)
        return attn_output, None

    attn.forward = cp_attention_forward


def apply_cp_to_model(model: nn.Module, parallel_config: dict) -> None:
    """Patch all attention modules in the model for context parallelism."""
    cp_group = parallel_config["cp_group"]
    cp_rank = parallel_config["cp_rank"]
    cp_size = parallel_config["cp_size"]

    patched = 0
    for module in model.modules():
        module_name = type(module).__name__
        # Match LFM2 attention classes (dense and MoE variants)
        if "Attention" in module_name and hasattr(module, "q_proj"):
            patch_attention_for_cp(module, cp_group, cp_rank, cp_size)
            patched += 1

    if patched == 0:
        logger.warning(
            "No attention modules found to patch for CP. "
            "Check that the model has modules with 'Attention' in the class name "
            "and q_proj attribute."
        )
    else:
        logger.info(f"Applied CP to {patched} attention modules (cp_size={cp_size})")


# === Sequence Splitting ===


def split_batch_for_cp(batch: dict, cp_rank: int, cp_size: int) -> dict:
    """Split input_ids, labels, attention_mask along sequence dim for this CP rank."""
    new_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.dim() >= 2:
            seq_len = value.shape[1]
            chunk_size = seq_len // cp_size
            start = cp_rank * chunk_size
            end = start + chunk_size
            new_batch[key] = value[:, start:end]
        else:
            new_batch[key] = value
    return new_batch


# === CP Loss Aggregation ===


def aggregate_cp_loss(
    loss: torch.Tensor, cp_group: dist.ProcessGroup, cp_size: int
) -> torch.Tensor:
    """Average loss across CP ranks (each rank computes loss on different sequence chunk)."""
    dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=cp_group)
    return loss / cp_size


# === Validation ===


def validate_cp_config(
    cp_size: int,
    max_length: int | None = None,
    world_size: int | None = None,
    ep_size: int = 1,
) -> None:
    """Validate context parallelism configuration."""
    if cp_size < 1:
        raise ValueError(f"cp_size must be >= 1, got {cp_size}")

    if max_length is not None and max_length % cp_size != 0:
        raise ValueError(
            f"max_length ({max_length}) must be divisible by cp_size ({cp_size})"
        )

    if world_size is not None:
        mesh_size = cp_size * ep_size
        if world_size % mesh_size != 0:
            raise ValueError(
                f"world_size ({world_size}) must be divisible by "
                f"cp_size * ep_size ({cp_size} * {ep_size} = {mesh_size})"
            )

    if cp_size > 1:
        logger.warning(
            "Context parallelism enabled — forcing SDPA attention "
            "(flash attention does not support custom causal masks)"
        )
