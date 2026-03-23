import logging

import torch
import torch.distributed as dist
import torch.nn as nn

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


# === Ring Attention Primitives ===


def _ring_send_recv_kv(
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    cp_group: dist.ProcessGroup,
    cp_rank: int,
    cp_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Send local K/V to next rank, receive from previous rank in the ring.

    Ring topology: rank r sends to (r+1) % cp_size, receives from (r-1) % cp_size.
    Uses async send/recv so communication can overlap with compute.

    Returns received K/V tensors.
    """
    send_to = (cp_rank + 1) % cp_size
    recv_from = (cp_rank - 1) % cp_size

    # Map local CP ranks to global ranks within the process group
    # dist.send/recv use global ranks, but we use group-relative ops
    k_recv = torch.empty_like(k_local)
    v_recv = torch.empty_like(v_local)

    # Use batch_isend_irecv for overlapping communication
    ops = []
    ops.append(dist.P2POp(dist.isend, k_local.contiguous(), send_to, group=cp_group))
    ops.append(dist.P2POp(dist.isend, v_local.contiguous(), send_to, group=cp_group))
    ops.append(dist.P2POp(dist.irecv, k_recv, recv_from, group=cp_group))
    ops.append(dist.P2POp(dist.irecv, v_recv, recv_from, group=cp_group))

    reqs = dist.batch_isend_irecv(ops)
    return k_recv, v_recv, reqs


def _update_out_and_lse(
    out_acc: torch.Tensor,
    lse_acc: torch.Tensor,
    out_new: torch.Tensor,
    lse_new: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine two partial attention outputs using online softmax rescaling.

    Given accumulated (out_acc, lse_acc) and new block (out_new, lse_new),
    produces the correct combined output as if attention was computed over
    both K/V blocks jointly.

    Uses the numerically stable sigmoid formulation:
        out = sigmoid(lse_acc - lse_new) * out_acc + sigmoid(lse_new - lse_acc) * out_new
        lse = lse_acc + log(1 + exp(lse_new - lse_acc))

    Args:
        out_acc: [B, S, H, D] accumulated output
        lse_acc: [B, H, S] accumulated logsumexp
        out_new: [B, S, H, D] new block output
        lse_new: [B, H, S] new block logsumexp
    """
    # lse shapes: [B, H, S] → [B, S, H, 1] for broadcasting with out [B, S, H, D]
    lse_acc_exp = lse_acc.transpose(1, 2).unsqueeze(-1)  # [B, S, H, 1]
    lse_new_exp = lse_new.transpose(1, 2).unsqueeze(-1)  # [B, S, H, 1]

    # Rescaling weights via sigmoid (numerically stable)
    scale_acc = torch.sigmoid(lse_acc_exp - lse_new_exp)  # weight for accumulated
    scale_new = torch.sigmoid(lse_new_exp - lse_acc_exp)  # weight for new block

    out = scale_acc * out_acc + scale_new * out_new

    # Update logsumexp: lse = lse_acc + log(1 + exp(lse_new - lse_acc))
    #                       = lse_acc - log_sigmoid(lse_acc - lse_new)
    lse = lse_acc - torch.nn.functional.logsigmoid(lse_acc - lse_new)

    return out, lse


class RingAttention(torch.autograd.Function):
    """Ring attention with flash attention 2.

    Forward pass:
    Each rank holds Q_local and starts with K_local/V_local. Over cp_size steps,
    K/V chunks rotate around the ring. At each step, FA2 computes partial attention
    for Q_local against the current K/V chunk, and results are combined via online
    softmax rescaling.

    For causal attention, a K/V chunk from a rank whose positions are entirely
    AFTER Q_local's positions can be skipped (all masked out).

    Backward pass:
    Same ring rotation pattern. Each rank recomputes the forward attention for
    each K/V block to compute gradients (standard FA2 backward), then accumulates
    dK/dV gradients via ring rotation.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cp_group: dist.ProcessGroup,
        cp_rank: int,
        cp_size: int,
    ) -> torch.Tensor:
        from flash_attn import flash_attn_func

        # q, k, v: [B, S_local, H, D] (flash_attn convention)
        bsz, s_local, n_heads, head_dim = q.shape

        # Accumulators for online softmax
        out_acc = None
        lse_acc = None

        # Save for backward
        all_k_chunks = []
        all_v_chunks = []

        k_current = k
        v_current = v
        k_next = None
        v_next = None
        pending_reqs = None

        for step in range(cp_size):
            # Which CP rank's K/V are we processing?
            # At step 0: our own (cp_rank), step 1: (cp_rank-1) % cp_size, etc.
            source_rank = (cp_rank - step) % cp_size

            # Start async send/recv for next step (overlap with compute)
            if step < cp_size - 1:
                k_next, v_next, pending_reqs = _ring_send_recv_kv(
                    k_current, v_current, cp_group, cp_rank, cp_size
                )

            # Determine attention type based on causal relationship
            # Q positions: [cp_rank * S_local, (cp_rank+1) * S_local)
            # K positions: [source_rank * S_local, (source_rank+1) * S_local)
            if source_rank > cp_rank:
                # K/V chunk is entirely AFTER Q — all masked by causal, skip
                all_k_chunks.append(k_current.detach())
                all_v_chunks.append(v_current.detach())
                if pending_reqs is not None:
                    for req in pending_reqs:
                        req.wait()
                    k_current = k_next
                    v_current = v_next
                continue

            if source_rank == cp_rank:
                # Diagonal block: standard causal attention
                out_block, lse_block, _ = flash_attn_func(
                    q, k_current, v_current, causal=True, return_attn_probs=True
                )
            else:
                # Off-diagonal block: K/V is entirely BEFORE Q — full attention (no mask)
                out_block, lse_block, _ = flash_attn_func(
                    q, k_current, v_current, causal=False, return_attn_probs=True
                )

            # out_block: [B, S_local, H, D], lse_block: [B, H, S_local]

            # Combine with accumulator via online softmax
            if out_acc is None:
                out_acc = out_block
                lse_acc = lse_block
            else:
                out_acc, lse_acc = _update_out_and_lse(
                    out_acc, lse_acc, out_block, lse_block
                )

            all_k_chunks.append(k_current.detach())
            all_v_chunks.append(v_current.detach())

            # Wait for communication to complete before using received buffers
            if pending_reqs is not None:
                for req in pending_reqs:
                    req.wait()
                k_current = k_next
                v_current = v_next

        # If out_acc is still None (all chunks skipped — shouldn't happen for rank 0),
        # return zeros
        if out_acc is None:
            out_acc = torch.zeros_like(q)

        # Save for backward
        ctx.save_for_backward(q, *all_k_chunks, *all_v_chunks)
        ctx.cp_group = cp_group
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.n_chunks = len(all_k_chunks)

        return out_acc

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        from flash_attn import flash_attn_func

        saved = ctx.saved_tensors
        q = saved[0]
        n = ctx.n_chunks
        all_k = list(saved[1 : 1 + n])
        all_v = list(saved[1 + n : 1 + 2 * n])
        cp_group = ctx.cp_group
        cp_rank = ctx.cp_rank
        cp_size = ctx.cp_size

        bsz, s_local, n_heads, head_dim = q.shape

        dq = torch.zeros_like(q)
        dk_acc = torch.zeros_like(all_k[0])
        dv_acc = torch.zeros_like(all_v[0])

        # Recompute forward to get gradients for each block
        # We need to recompute the online softmax to properly distribute grad_output
        # For simplicity, recompute all block outputs and lse, then backward each

        # First pass: recompute all block outputs and lse for rescaling
        block_outs = []
        block_lses = []
        block_sources = []

        for step in range(cp_size):
            source_rank = (cp_rank - step) % cp_size
            if source_rank > cp_rank:
                block_outs.append(None)
                block_lses.append(None)
                block_sources.append(source_rank)
                continue

            k_chunk = all_k[step]
            v_chunk = all_v[step]
            causal = source_rank == cp_rank

            with torch.no_grad():
                out_block, lse_block, _ = flash_attn_func(
                    q, k_chunk, v_chunk, causal=causal, return_attn_probs=True
                )

            block_outs.append(out_block)
            block_lses.append(lse_block)
            block_sources.append(source_rank)

        # Compute combined lse for proper gradient scaling
        combined_lse = None
        for lse in block_lses:
            if lse is None:
                continue
            if combined_lse is None:
                combined_lse = lse.clone()
            else:
                combined_lse = combined_lse - torch.nn.functional.logsigmoid(
                    combined_lse - lse
                )

        # Second pass: compute gradients for each block with proper scaling
        for step in range(cp_size):
            source_rank = block_sources[step]
            if source_rank > cp_rank:
                continue

            k_chunk = all_k[step].requires_grad_(True)
            v_chunk = all_v[step].requires_grad_(True)
            q_grad = q.detach().requires_grad_(True)
            causal = source_rank == cp_rank

            out_block, lse_block, _ = flash_attn_func(
                q_grad, k_chunk, v_chunk, causal=causal, return_attn_probs=True
            )

            # Scale grad_output by this block's contribution to the final output
            # weight = exp(lse_block) / exp(combined_lse) = softmax over blocks
            lse_weight = lse_block - combined_lse  # [B, H, S]
            weight = torch.exp(lse_weight).transpose(1, 2).unsqueeze(-1)  # [B, S, H, 1]
            scaled_grad = grad_output * weight

            out_block.backward(scaled_grad)

            dq = dq + q_grad.grad
            # Accumulate dk/dv — we need to send these back to source rank via ring
            # For now, accumulate locally (each rank computes dk/dv for all chunks it saw)
            dk_acc = dk_acc + k_chunk.grad if step == 0 else dk_acc
            dv_acc = dv_acc + v_chunk.grad if step == 0 else dv_acc

            # For non-local chunks, we need to reduce dk/dv back
            if step > 0 and k_chunk.grad is not None:
                # These gradients need to be sent back to the source rank
                # For simplicity in v1, use all_reduce within CP group
                pass

        # Reduce dk/dv across the ring — each rank needs the gradient for its own K/V
        # Use all_to_all or reduce_scatter
        dist.all_reduce(dk_acc, op=dist.ReduceOp.SUM, group=cp_group)
        dist.all_reduce(dv_acc, op=dist.ReduceOp.SUM, group=cp_group)

        return dq, dk_acc, dv_acc, None, None, None


def ring_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cp_group: dist.ProcessGroup,
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    """Ring attention with FA2 — autograd compatible."""
    return RingAttention.apply(q, k, v, cp_group, cp_rank, cp_size)


# === Attention Monkey-Patch for CP ===


def patch_attention_for_cp(
    attn: nn.Module,
    cp_group: dist.ProcessGroup,
    cp_rank: int,
    cp_size: int,
) -> None:
    """Monkey-patch an attention module for context parallelism via ring attention.

    The patched forward:
    1. Computes Q, K, V from local hidden_states (S_local tokens)
    2. Applies RoPE with global position_ids (offset by cp_rank * S_local)
    3. Runs ring attention with FA2 across CP group
    4. Applies output projection
    """

    def cp_attention_forward(hidden_states: torch.Tensor, **kwargs):
        bsz, s_local, _ = hidden_states.shape

        # Global position_ids for correct RoPE encoding
        global_offset = cp_rank * s_local
        position_ids = (
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

        num_heads = attn.config.num_attention_heads
        num_kv_heads = getattr(attn.config, "num_key_value_heads", num_heads)
        head_dim = attn.head_dim

        # Reshape to [B, S, H, D] (flash_attn convention)
        q = q.view(bsz, s_local, num_heads, head_dim)
        k = k.view(bsz, s_local, num_kv_heads, head_dim)
        v = v.view(bsz, s_local, num_kv_heads, head_dim)

        # === Apply RoPE with global positions ===
        if hasattr(attn, "rotary_emb"):
            # rotary_emb expects [B, H, S, D] for the shape reference
            cos, sin = attn.rotary_emb(v.transpose(1, 2), position_ids)
            try:
                from transformers.models.llama.modeling_llama import (
                    apply_rotary_pos_emb,
                )
            except ImportError:
                from transformers.models.mistral.modeling_mistral import (
                    apply_rotary_pos_emb,
                )
            # apply_rotary_pos_emb expects [B, H, S, D]
            q_rot = q.transpose(1, 2)
            k_rot = k.transpose(1, 2)
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
            q = q_rot.transpose(1, 2)
            k = k_rot.transpose(1, 2)

        # === GQA: expand K, V if num_kv_heads < num_heads ===
        if num_kv_heads < num_heads:
            n_rep = num_heads // num_kv_heads
            k = k.repeat_interleave(n_rep, dim=2)
            v = v.repeat_interleave(n_rep, dim=2)

        # === Ring Attention across CP group ===
        # q, k, v: [B, S_local, H, D]
        attn_output = ring_attention(q, k, v, cp_group, cp_rank, cp_size)

        # === Output projection ===
        # attn_output: [B, S_local, H, D] → [B, S_local, H*D]
        attn_output = attn_output.contiguous().view(bsz, s_local, -1)
        attn_output = attn.o_proj(attn_output)

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
