import logging
import socket

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from leap_finetune.utils.parallel_topology import (
    WorkerTopology,
    build_node_local_blocks,
)

logger = logging.getLogger(__name__)

CP_CONV_LEFT_HALO = 2


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
        "dp_rank": dp_idx,
    }

    logger.info("Rank %s: dp=%s cp=%s/%s", rank, dp_idx, cp_idx, cp_size)
    return result


def _get_cp_contiguous_bounds(
    seq_len: int,
    cp_rank: int,
    cp_size: int,
) -> tuple[int, int]:
    """Return the contiguous sequence bounds for one CP rank."""
    if seq_len % cp_size != 0:
        raise ValueError(
            f"Sequence length {seq_len} must be divisible by cp_size {cp_size}"
        )
    chunk_len = seq_len // cp_size
    start = cp_rank * chunk_len
    end = start + chunk_len
    return start, end


def _all_gather_fixed_seq(
    tensor: torch.Tensor,
    group: dist.ProcessGroup,
) -> list[torch.Tensor]:
    """All-gather equal-length sequence shards across the CP group."""
    return [
        chunk.contiguous() for chunk in _CPAllGather.apply(tensor.contiguous(), group)
    ]


class _CPAllGather(torch.autograd.Function):
    """Differentiable all-gather with an explicit per-chunk gradient reduction."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: dist.ProcessGroup):
        ctx.group = group
        ctx.rank = dist.get_rank(group=group)
        world_size = dist.get_world_size(group=group)
        chunks = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(chunks, tensor.contiguous(), group=group)
        return tuple(chunks)

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor):
        grad_stack = torch.stack(
            [grad.contiguous() for grad in grad_outputs],
            dim=0,
        )
        dist.all_reduce(grad_stack, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_stack[ctx.rank].contiguous(), None


def prefix_gather_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cp_group: dist.ProcessGroup,
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    """Run causal attention over the contiguous visible prefix for this CP rank."""
    from flash_attn import flash_attn_func

    if cp_size == 1:
        return flash_attn_func(q, k, v, causal=True)

    gathered_k_chunks = _all_gather_fixed_seq(k, cp_group)
    gathered_v_chunks = _all_gather_fixed_seq(v, cp_group)
    visible_k = torch.cat(gathered_k_chunks[: cp_rank + 1], dim=1).contiguous()
    visible_v = torch.cat(gathered_v_chunks[: cp_rank + 1], dim=1).contiguous()
    return flash_attn_func(q, visible_k, visible_v, causal=True)


def _all_gather_cp_halos(
    tensor: torch.Tensor,
    halo_width: int,
    group: dist.ProcessGroup,
) -> list[torch.Tensor]:
    """All-gather a fixed-width tail halo from every CP rank."""
    if halo_width < 1:
        raise ValueError(f"halo_width must be >= 1, got {halo_width}")
    if tensor.shape[1] < halo_width:
        raise ValueError(
            f"Need sequence length >= halo width ({halo_width}), got {tensor.shape[1]}"
        )
    tail = tensor[:, -halo_width:].contiguous()
    return _all_gather_fixed_seq(tail, group)


def _prepend_cp_left_halo(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    cp_group: dist.ProcessGroup,
    cp_rank: int,
    halo_width: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Prepend the previous rank's tail halo to a local conv input shard."""
    if halo_width < 1:
        return hidden_states, attention_mask

    gathered_hidden_halos = _all_gather_cp_halos(hidden_states, halo_width, cp_group)
    if cp_rank == 0:
        halo_hidden = torch.zeros_like(gathered_hidden_halos[0])
    else:
        halo_hidden = gathered_hidden_halos[cp_rank - 1]
    extended_hidden = torch.cat((halo_hidden, hidden_states), dim=1)

    extended_mask = attention_mask
    if attention_mask is not None:
        gathered_mask_halos = _all_gather_cp_halos(attention_mask, halo_width, cp_group)
        if cp_rank == 0:
            halo_mask = torch.zeros_like(gathered_mask_halos[0])
        else:
            halo_mask = gathered_mask_halos[cp_rank - 1]
        extended_mask = torch.cat((halo_mask, attention_mask), dim=1)

    return extended_hidden, extended_mask


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
            start = cp_rank * s_local if cp_size > 1 else 0
            position_ids = (
                torch.arange(
                    start,
                    start + s_local,
                    device=hidden_states.device,
                    dtype=torch.long,
                )
                .unsqueeze(0)
                .expand(bsz, -1)
            )

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
        elif "ShortConv" in module_name and hasattr(module, "conv"):
            patch_short_conv_for_cp(module, cp_group, cp_rank, cp_size)
            patched += 1

    if patched == 0:
        logger.warning(
            "No attention modules found to patch for CP. "
            "Check that the model has modules with 'Attention' in the class name "
            "and q_proj attribute."
        )
    else:
        logger.info(f"Applied CP to {patched} attention modules (cp_size={cp_size})")


def patch_short_conv_for_cp(
    conv_module: nn.Module,
    cp_group: dist.ProcessGroup,
    cp_rank: int,
    cp_size: int,
) -> None:
    """Monkey-patch the LFM short-conv path to prepend a minimal left halo."""
    original_forward = conv_module.forward
    halo_width = min(
        getattr(conv_module, "L_cache", CP_CONV_LEFT_HALO) - 1, CP_CONV_LEFT_HALO
    )

    def cp_short_conv_forward(
        hidden_states: torch.Tensor,
        past_key_values=None,
        cache_position=None,
        attention_mask=None,
        *args,
        **kwargs,
    ):
        if cp_size == 1 or halo_width < 1:
            return original_forward(
                hidden_states,
                past_key_values=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
                *args,
                **kwargs,
            )

        # Keep generation/cache behavior untouched; CP is only used for training/eval.
        if past_key_values is not None or cache_position is not None:
            return original_forward(
                hidden_states,
                past_key_values=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
                *args,
                **kwargs,
            )

        extended_hidden, extended_mask = _prepend_cp_left_halo(
            hidden_states,
            attention_mask,
            cp_group=cp_group,
            cp_rank=cp_rank,
            halo_width=halo_width,
        )
        conv_out = original_forward(
            extended_hidden,
            past_key_values=None,
            cache_position=None,
            attention_mask=extended_mask,
            *args,
            **kwargs,
        )
        return conv_out[:, halo_width:, :].contiguous()

    conv_module.forward = cp_short_conv_forward


# === Sequence Splitting ===


def _pad_sequence_tensor_for_cp(
    key: str, value: torch.Tensor, cp_size: int
) -> torch.Tensor:
    """Pad the sequence dimension to a CP-compatible multiple when needed."""
    seq_len = value.shape[1]
    pad_multiple = cp_size if cp_size > 1 else 1
    pad_len = (-seq_len) % pad_multiple
    if pad_len == 0:
        return value

    pad_shape = list(value.shape)
    pad_shape[1] = pad_len

    if key in {"labels", "shift_labels"}:
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

    labels = batch_with_positions.get("labels")
    if isinstance(labels, torch.Tensor) and labels.dim() >= 2:
        padded_labels = _pad_sequence_tensor_for_cp("labels", labels, cp_size)
        shifted_labels = torch.full_like(padded_labels, -100)
        shifted_labels[:, :-1, ...] = padded_labels[:, 1:, ...]
        batch_with_positions["labels"] = padded_labels
        batch_with_positions["shift_labels"] = shifted_labels

    new_batch = {}
    for key, value in batch_with_positions.items():
        if isinstance(value, torch.Tensor) and value.dim() >= 2:
            # === 2. Pad every sequence-like tensor to a CP-compatible multiple ===
            value = _pad_sequence_tensor_for_cp(key, value, cp_size)
            seq_len = value.shape[1]
            if cp_size > 1:
                start, end = _get_cp_contiguous_bounds(seq_len, cp_rank, cp_size)
                new_batch[key] = value[:, start:end, ...].contiguous()
            else:
                new_batch[key] = value
        else:
            new_batch[key] = value

    attention_mask = new_batch.get("attention_mask")
    if isinstance(attention_mask, torch.Tensor):
        new_batch["leap_cp_padding_mask"] = attention_mask.clone()
    return new_batch


def _cp_tensor_fingerprint(tensor: torch.Tensor) -> tuple:
    tensor = tensor.detach()
    flat = tensor.reshape(-1)
    sample_len = min(8, flat.numel())
    head = flat[:sample_len].to("cpu").tolist()
    tail = flat[-sample_len:].to("cpu").tolist() if sample_len else []
    checksum = tensor.to(torch.int64).sum().item() if tensor.numel() else 0
    return (tuple(tensor.shape), int(checksum), tuple(head), tuple(tail))


def validate_cp_batch_replicated(
    batch: dict,
    cp_group: dist.ProcessGroup,
    cp_rank: int,
    cp_size: int,
) -> None:
    """Fail fast if CP peers did not receive the same full batch before splitting."""
    fingerprint = {}
    for key in ("input_ids", "labels", "attention_mask"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            fingerprint[key] = _cp_tensor_fingerprint(value)

    gathered: list[dict | None] = [None] * cp_size
    dist.all_gather_object(gathered, fingerprint, group=cp_group)

    expected = gathered[0]
    mismatched = [rank for rank, item in enumerate(gathered) if item != expected]
    if mismatched:
        raise RuntimeError(
            "Context parallel ranks received different pre-split batches. "
            f"cp_rank={cp_rank}, mismatched_cp_ranks={mismatched}, "
            f"fingerprints={gathered}"
        )


# === CP Loss Aggregation ===


def aggregate_cp_loss(
    loss: torch.Tensor, cp_group: dist.ProcessGroup, cp_size: int
) -> torch.Tensor:
    """Average the reporting loss across CP ranks."""
    dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=cp_group)
    return loss / cp_size


def validate_cp_model_support(model: nn.Module, train_config: dict) -> None:
    """Reject known-unsafe CP combinations while keeping the baseline runtime shape."""
    if not train_config.get("packing"):
        return

    layer_types = getattr(getattr(model, "config", None), "layer_types", None) or []
    has_conv_layers = any(
        "conv" in str(layer_type).lower() for layer_type in layer_types
    )
    if not has_conv_layers:
        has_conv_layers = any(
            "conv" in type(module).__name__.lower() for module in model.modules()
        )
    if has_conv_layers:
        raise ValueError(
            "packing=True is not supported with context_parallel_size > 1 on "
            "hybrid/conv LFM models. Packed sequence boundaries can bleed into "
            "short-conv state under CP."
        )


def compute_cp_causal_lm_loss(
    model: nn.Module,
    inputs: dict,
    cp_group: dist.ProcessGroup,
    cp_size: int,
    return_outputs: bool = False,
    num_items_in_batch: torch.Tensor | int | float | None = None,
):
    """Compute a CP-local causal LM loss that remains differentiable on empty-label shards."""
    labels = inputs.get("labels")
    if labels is None:
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        if return_outputs:
            return loss, outputs
        return loss

    model_inputs = dict(inputs)
    model_inputs.pop("labels")
    shift_labels = model_inputs.pop("shift_labels", None)
    outputs = model(**model_inputs)
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits

    if shift_labels is None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
    else:
        shift_logits = logits.contiguous()
        shift_labels = shift_labels.contiguous()
    vocab_size = shift_logits.size(-1)

    token_losses = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)
    valid_mask = shift_labels.ne(-100)
    local_token_count = valid_mask.sum()

    if bool(local_token_count.item()):
        local_loss_numerator = token_losses.masked_select(valid_mask).sum()
    else:
        # Preserve a gradient-connected zero so backward is well-defined on shards
        # that contain no supervised assistant tokens after CP splitting.
        local_loss_numerator = shift_logits.sum() * 0.0

    global_token_count = local_token_count.to(
        device=shift_logits.device, dtype=local_loss_numerator.dtype
    )
    dist.all_reduce(global_token_count, op=dist.ReduceOp.SUM, group=cp_group)

    if num_items_in_batch is not None:
        denominator = torch.as_tensor(
            num_items_in_batch,
            device=shift_logits.device,
            dtype=local_loss_numerator.dtype,
        )
    else:
        denominator = global_token_count

    if bool(denominator.item()):
        # Distributed reducers average gradients across ranks. CP ranks each hold
        # a chunk of the same sequence, so scale the local objective before
        # backward; the reducer average then recovers the full-sequence gradient.
        loss = (local_loss_numerator / denominator) * cp_size
    else:
        loss = shift_logits.sum() * 0.0

    if isinstance(outputs, dict):
        outputs["loss"] = loss
    else:
        outputs.loss = loss
    if return_outputs:
        return loss, outputs
    return loss


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
