import logging
import socket

import torch
import torch.distributed as dist
import torch.nn as nn

from leap_finetune.utils.parallel_topology import (
    WorkerTopology,
    build_node_local_blocks,
)

logger = logging.getLogger(__name__)


def dist_barrier(group: dist.ProcessGroup | None = None) -> None:
    """Run a NCCL-safe barrier on the current CUDA device when available."""
    if not dist.is_available() or not dist.is_initialized():
        return

    barrier_kwargs = {}
    if torch.cuda.is_available():
        barrier_kwargs["device_ids"] = [torch.cuda.current_device()]

    if group is None:
        dist.barrier(**barrier_kwargs)
    else:
        dist.barrier(group=group, **barrier_kwargs)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by half for RoPE application."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q/K using the standard HF formulation."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _is_position_embedding_pair(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and all(torch.is_tensor(item) for item in value)
    )


def _get_rank_topology() -> list[dict[str, int | str]]:
    """Collect host metadata for node-local CP groups."""
    gathered: list[dict[str, int | str] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(
        gathered,
        {
            "rank": dist.get_rank(),
            "host": socket.gethostname(),
        },
    )
    return [item for item in gathered if item is not None]


def _build_topology_local_blocks(
    cp_size: int,
) -> list[tuple[str, list[int], int]]:
    """Build node-local CP blocks instead of assuming global rank contiguity."""
    topology = _get_rank_topology()
    workers = [
        WorkerTopology(
            rank=int(item["rank"]),
            node_id=str(item["host"]),
            local_order=int(item["rank"]),
        )
        for item in topology
    ]
    return build_node_local_blocks(workers, cp_size)


# === CP Process Groups ===


def create_parallel_process_groups(cp_size: int = 1) -> dict:
    """Create node-local CP groups."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size % cp_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by cp_size ({cp_size})"
        )

    blocks = _build_topology_local_blocks(cp_size)

    my_cp_group = None
    dp_idx = cp_idx = None

    dist_barrier()
    for block_index, (host, block_ranks, local_dp_idx) in enumerate(blocks):
        group = dist.new_group(block_ranks)
        if rank in block_ranks:
            my_cp_group = group
            cp_idx = block_ranks.index(rank)
            dp_idx = block_index
            logger.info(
                "Rank %s: node-local CP group ranks=%s host=%s local_dp=%s",
                rank,
                block_ranks,
                host,
                local_dp_idx,
            )

    dist_barrier()

    if my_cp_group is None or cp_idx is None:
        raise RuntimeError(f"Rank {rank} failed to join a CP group")

    result = {
        "cp_group": my_cp_group,
        "cp_rank": cp_idx,
        "cp_size": cp_size,
    }

    logger.info("Rank %s: dp=%s cp=%s/%s", rank, dp_idx, cp_idx, cp_size)
    return result


def _get_cp_load_balanced_chunk_ids(
    cp_rank: int, cp_size: int, device: torch.device
) -> torch.Tensor:
    """Return the two sequence chunk ids assigned to this CP rank."""
    return torch.tensor(
        [cp_rank, 2 * cp_size - cp_rank - 1], device=device, dtype=torch.long
    )


def _all_gather_variable_seq(
    tensor: torch.Tensor,
    group: dist.ProcessGroup,
) -> tuple[list[torch.Tensor], list[int]]:
    """All-gather sequence shards after padding them to a CP-group max length."""
    from torch.distributed.nn import functional as dist_nn

    local_len = torch.tensor([tensor.shape[1]], device=tensor.device, dtype=torch.int64)
    gathered_lens = [torch.zeros_like(local_len) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered_lens, local_len, group=group)
    lengths = [int(item.item()) for item in gathered_lens]
    max_len = max(lengths)

    if tensor.shape[1] < max_len:
        pad_shape = list(tensor.shape)
        pad_shape[1] = max_len - tensor.shape[1]
        pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat((tensor, pad), dim=1)

    gathered = dist_nn.all_gather(tensor.contiguous(), group=group)
    trimmed = [chunk[:, :seq_len].contiguous() for chunk, seq_len in zip(gathered, lengths)]
    return trimmed, lengths


def prefix_gather_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cp_group: dist.ProcessGroup,
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    """Run causal attention with staged low/high-side CP gathers."""
    from flash_attn import flash_attn_func

    if cp_size == 1:
        return flash_attn_func(q, k, v, causal=True)

    bsz, s_local, _, _ = q.shape
    subchunk_len = s_local // 2
    local_chunk_ids = _get_cp_load_balanced_chunk_ids(cp_rank, cp_size, device=q.device)

    q_low = q[:, :subchunk_len].contiguous()
    q_high = q[:, subchunk_len:].contiguous()
    k_low = k[:, :subchunk_len].contiguous()
    k_high = k[:, subchunk_len:].contiguous()
    v_low = v[:, :subchunk_len].contiguous()
    v_high = v[:, subchunk_len:].contiguous()

    # === 1. Gather the early chunks [0, ..., cp_size - 1] across the CP group ===
    gathered_low_k_chunks, low_lengths = _all_gather_variable_seq(k_low, cp_group)
    gathered_low_v_chunks, _ = _all_gather_variable_seq(v_low, cp_group)
    gathered_low_k = torch.cat(gathered_low_k_chunks, dim=1)
    gathered_low_v = torch.cat(gathered_low_v_chunks, dim=1)

    # === 2. Run the first local subchunk as soon as its visible prefix is available ===
    low_chunk_id = int(local_chunk_ids[0].item())
    low_prefix_end = sum(low_lengths[: low_chunk_id + 1])
    low_out = flash_attn_func(
        q_low,
        gathered_low_k[:, :low_prefix_end].contiguous(),
        gathered_low_v[:, :low_prefix_end].contiguous(),
        causal=True,
    )

    # === 3. Gather the late chunks [cp_size, ..., 2 * cp_size - 1] in a second round ===
    gathered_high_k_chunks, high_lengths = _all_gather_variable_seq(k_high, cp_group)
    gathered_high_v_chunks, _ = _all_gather_variable_seq(v_high, cp_group)
    gathered_high_k_chunks.reverse()
    gathered_high_v_chunks.reverse()
    high_lengths.reverse()
    gathered_high_k = torch.cat(gathered_high_k_chunks, dim=1)
    gathered_high_v = torch.cat(gathered_high_v_chunks, dim=1)

    # === 4. Run the second local subchunk against the full visible prefix ===
    high_chunk_id = int(local_chunk_ids[1].item())
    high_prefix_chunks = high_chunk_id - cp_size + 1
    high_prefix_end = sum(high_lengths[:high_prefix_chunks])
    high_prefix_k = torch.cat(
        (
            gathered_low_k,
            gathered_high_k[:, :high_prefix_end].contiguous(),
        ),
        dim=1,
    )
    high_prefix_v = torch.cat(
        (
            gathered_low_v,
            gathered_high_v[:, :high_prefix_end].contiguous(),
        ),
        dim=1,
    )
    high_out = flash_attn_func(q_high, high_prefix_k, high_prefix_v, causal=True)

    return torch.cat((low_out, high_out), dim=1)


# === Attention Monkey-Patch for CP ===


def patch_attention_for_cp(
    attn: nn.Module,
    cp_group: dist.ProcessGroup,
    cp_rank: int,
    cp_size: int,
    rotary_emb: nn.Module | None = None,
) -> None:
    """Monkey-patch an attention module for context parallelism."""
    out_proj = getattr(attn, "out_proj", None)
    if out_proj is None:
        raise AttributeError(f"{type(attn).__name__} is missing required 'out_proj'")

    # Resolve rotary embedding: on the attention module or passed from model
    rope = getattr(attn, "rotary_emb", None) or rotary_emb

    def cp_attention_forward(hidden_states: torch.Tensor, *args, **kwargs):
        # Ensure dtype matches model weights (DeepSpeed may pass float32 inputs)
        param_dtype = attn.q_proj.weight.dtype
        if hidden_states.dtype != param_dtype:
            hidden_states = hidden_states.to(param_dtype)

        bsz, s_local, _ = hidden_states.shape

        position_ids = kwargs.get("position_ids")
        position_embeddings = kwargs.get("position_embeddings")
        remaining_args = list(args)

        if remaining_args:
            first_arg = remaining_args.pop(0)
            if _is_position_embedding_pair(first_arg):
                position_embeddings = position_embeddings or first_arg

        if remaining_args:
            second_arg = remaining_args.pop(0)
            if _is_position_embedding_pair(second_arg):
                position_embeddings = position_embeddings or second_arg
            elif torch.is_tensor(second_arg) and position_ids is None:
                position_ids = second_arg

        if position_ids is None:
            subchunk_len = s_local // 2 if cp_size > 1 else s_local
            local_chunk_ids = _get_cp_load_balanced_chunk_ids(
                cp_rank, cp_size, device=hidden_states.device
            )
            position_chunks = [
                torch.arange(
                    int(chunk_id.item()) * subchunk_len,
                    (int(chunk_id.item()) + 1) * subchunk_len,
                    device=hidden_states.device,
                    dtype=torch.long,
                )
                for chunk_id in local_chunk_ids
            ]
            position_ids = torch.cat(position_chunks, dim=0).unsqueeze(0).expand(bsz, -1)

        num_heads = attn.config.num_attention_heads
        num_kv_heads = getattr(attn.config, "num_key_value_heads", num_heads)
        head_dim = attn.head_dim
        q_shape = (bsz, s_local, num_heads, head_dim)
        kv_shape = (bsz, s_local, num_kv_heads, head_dim)

        # === Q, K, V projections ===
        q = attn.q_proj(hidden_states).view(*q_shape)
        k = attn.k_proj(hidden_states).view(*kv_shape)
        v = attn.v_proj(hidden_states).view(*kv_shape)

        # Apply Q/K layer norms if present (LFM2-style)
        if hasattr(attn, "q_layernorm"):
            q = attn.q_layernorm(q)
        if hasattr(attn, "k_layernorm"):
            k = attn.k_layernorm(k)

        # Transpose to [B, H, S, D] for RoPE
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # === Apply RoPE with global positions ===
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = _apply_rotary_pos_emb(q, k, cos, sin)
        elif rope is not None:
            cos, sin = rope(hidden_states, position_ids)
            q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        # Back to [B, S, H, D] for flash_attn
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        # === GQA: expand K, V if num_kv_heads < num_heads ===
        if num_kv_heads < num_heads:
            n_rep = num_heads // num_kv_heads
            k = k.repeat_interleave(n_rep, dim=2)
            v = v.repeat_interleave(n_rep, dim=2)

        # === 1. Gather/reorder the CP K/V chunks and run prefix attention per subchunk ===
        attn_output = prefix_gather_attention(q, k, v, cp_group, cp_rank, cp_size)

        # === 2. Project back to the model width ===
        # [B, S_local, H, D] → [B, S_local, H*D]
        attn_output = attn_output.contiguous().view(bsz, s_local, -1)
        proj_dtype = out_proj.weight.dtype
        if attn_output.dtype != proj_dtype:
            attn_output = attn_output.to(proj_dtype)
        attn_output = out_proj(attn_output)

        # HF attention returns (attn_output, attn_weights, past_key_value)
        return attn_output, None

    attn.forward = cp_attention_forward


def apply_cp_to_model(model: nn.Module, parallel_config: dict) -> None:
    """Patch all attention modules in the model for context parallelism."""
    cp_group = parallel_config["cp_group"]
    cp_rank = parallel_config["cp_rank"]
    cp_size = parallel_config["cp_size"]

    # Find model-level rotary embedding (LFM2 stores it as model.model.pos_emb)
    rotary_emb = None
    for name, module in model.named_modules():
        if "RotaryEmbedding" in type(module).__name__:
            rotary_emb = module
            logger.info(f"Found model-level rotary embedding at: {name}")
            break

    patched = 0
    for module in model.modules():
        module_name = type(module).__name__
        if "Attention" in module_name and hasattr(module, "q_proj"):
            patch_attention_for_cp(
                module,
                cp_group,
                cp_rank,
                cp_size,
                rotary_emb=rotary_emb,
            )
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


def _pad_sequence_tensor_for_cp(
    key: str, value: torch.Tensor, cp_size: int
) -> torch.Tensor:
    """Pad the sequence dimension to a CP-compatible multiple when needed."""
    seq_len = value.shape[1]
    pad_multiple = 2 * cp_size if cp_size > 1 else 1
    pad_len = (-seq_len) % pad_multiple
    if pad_len == 0:
        return value

    pad_shape = list(value.shape)
    pad_shape[1] = pad_len

    if key == "labels":
        pad_values = torch.full(
            pad_shape,
            fill_value=-100,
            dtype=value.dtype,
            device=value.device,
        )
    elif key == "position_ids":
        if value.dim() != 2:
            raise ValueError(
                f"Unsupported position_ids shape for CP padding: {tuple(value.shape)}"
            )
        start = seq_len
        stop = seq_len + pad_len
        pad_values = (
            torch.arange(start, stop, device=value.device, dtype=value.dtype)
            .unsqueeze(0)
            .expand(value.shape[0], -1)
        )
    else:
        pad_values = torch.zeros(pad_shape, dtype=value.dtype, device=value.device)

    logger.info(
        "Padding '%s' from sequence length %s to %s for cp_size=%s",
        key,
        seq_len,
        seq_len + pad_len,
        cp_size,
    )
    return torch.cat((value, pad_values), dim=1)


def split_batch_for_cp(batch: dict, cp_rank: int, cp_size: int) -> dict:
    """Split sequence-like batch tensors for the current CP rank.

    When `position_ids` are absent, inject global positions for standard LM inputs
    so model-level RoPE uses consistent absolute offsets across CP ranks.
    """
    # === 1. Materialize global positions before any CP-local reshaping ===
    batch_with_positions = dict(batch)

    input_ids = batch_with_positions.get("input_ids")
    if (
        "position_ids" not in batch_with_positions
        and isinstance(input_ids, torch.Tensor)
        and input_ids.dim() >= 2
    ):
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(
            seq_len, device=input_ids.device, dtype=torch.long
        ).unsqueeze(0)
        batch_with_positions["position_ids"] = position_ids.expand(
            input_ids.shape[0], -1
        )

    new_batch = {}
    for key, value in batch_with_positions.items():
        if isinstance(value, torch.Tensor) and value.dim() >= 2:
            # === 2. Pad every sequence-like tensor to a CP-compatible multiple ===
            value = _pad_sequence_tensor_for_cp(key, value, cp_size)
            seq_len = value.shape[1]
            if cp_size > 1:
                # === 3. Reassemble the load-balanced local shard for this CP rank ===
                subchunk_len = seq_len // (2 * cp_size)
                value = value.view(
                    value.shape[0],
                    2 * cp_size,
                    subchunk_len,
                    *value.shape[2:],
                )
                chunk_ids = _get_cp_load_balanced_chunk_ids(
                    cp_rank, cp_size, value.device
                )
                value = value.index_select(1, chunk_ids).reshape(
                    value.shape[0],
                    2 * subchunk_len,
                    *value.shape[3:],
                )
                new_batch[key] = value
            else:
                new_batch[key] = value
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
) -> None:
    """Validate context parallelism configuration. Requires flash-attn >= 2.8."""
    if cp_size < 1:
        raise ValueError(f"cp_size must be >= 1, got {cp_size}")

    if cp_size > 1:
        from transformers.utils import is_flash_attn_2_available

        if not is_flash_attn_2_available():
            raise RuntimeError(
                "Context parallelism requires flash-attn >= 2.8. "
                "Install with: pip install flash-attn>=2.8.0"
            )

    # Batch splitting pads sequence-like tensors to a CP-compatible multiple at runtime,
    # so max_length itself does not need to be divisible by cp_size.
    _ = max_length

    if world_size is not None and world_size % cp_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by cp_size ({cp_size})"
        )
