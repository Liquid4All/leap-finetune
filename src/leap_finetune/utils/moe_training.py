import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# === MoE Training Config ===


@dataclass
class MoETrainingConfig:
    aux_loss_coef: float = 0.01
    z_loss_coef: float = 0.001
    capacity_factor: float | None = None
    token_drop_policy: str = "probs"
    router_lr_ratio: float = 0.1

    @classmethod
    def from_dict(cls, d: dict) -> "MoETrainingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# === Auxiliary Loss Injection ===


class InjectAuxLoss(torch.autograd.Function):
    """Injects auxiliary loss into backward pass without affecting forward output.

    The aux loss is added to the gradient of the output during backward,
    which propagates it through the computation graph without changing
    the forward pass values.
    """

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (aux_loss,) = ctx.saved_tensors
        aux_loss_grad = torch.ones_like(aux_loss)
        # Scale aux loss gradient to match output shape — broadcast add to first element
        scaled_grad = torch.zeros_like(grad_output)
        scaled_grad.view(-1)[0] = aux_loss_grad.sum()
        return grad_output + scaled_grad, None


# === Loss Functions ===


def switch_load_balancing_loss(
    router_probs: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
    top_k: int,
    coeff: float,
) -> torch.Tensor:
    """Switch Transformer load balancing auxiliary loss.

    Encourages equal routing across experts by penalizing correlation
    between aggregated routing probabilities and actual token assignments.

    Args:
        router_probs: [n_tokens, n_experts] routing probabilities
        selected_experts: [n_tokens, top_k] expert indices
        num_experts: total number of experts
        top_k: number of experts per token
        coeff: loss coefficient
    """
    n_tokens = router_probs.shape[0]

    # tokens_per_expert: [n_experts] — count of tokens routed to each expert
    tokens_per_expert = torch.zeros(
        num_experts, device=router_probs.device, dtype=router_probs.dtype
    )
    ones = torch.ones_like(selected_experts, dtype=router_probs.dtype)
    tokens_per_expert.scatter_add_(0, selected_experts.view(-1), ones.view(-1))

    # aggregated_probs: [n_experts] — mean routing probability per expert
    aggregated_probs = router_probs.mean(dim=0)

    aux_loss = (
        torch.sum(aggregated_probs * tokens_per_expert)
        * num_experts
        * coeff
        / (n_tokens * top_k)
    )
    return aux_loss


def z_loss(router_logits: torch.Tensor, coeff: float) -> torch.Tensor:
    """ST-MoE z-loss — penalizes large router logits to prevent divergence.

    Args:
        router_logits: [n_tokens, n_experts] raw router logits (before softmax)
        coeff: loss coefficient
    """
    return coeff * router_logits.logsumexp(dim=-1).pow(2).mean()


# === Token Dropping ===


def apply_router_token_dropping(
    router_probs: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
    capacity_factor: float,
    drop_policy: str = "probs",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply capacity-based token dropping to routing decisions.

    Limits tokens per expert to capacity_factor * (n_tokens / n_experts).
    Tokens exceeding capacity are dropped (routing weight set to 0).

    Args:
        router_probs: [n_tokens, n_experts] routing probabilities
        selected_experts: [n_tokens, top_k] selected expert indices
        routing_weights: [n_tokens, top_k] routing weights
        num_experts: total number of experts
        capacity_factor: multiplier for expert capacity (e.g. 1.25)
        drop_policy: "probs" (drop lowest prob) or "position" (drop last)

    Returns:
        Updated (selected_experts, routing_weights) with dropped tokens zeroed out.
    """
    n_tokens, top_k = selected_experts.shape
    expert_capacity = int(capacity_factor * n_tokens / num_experts)

    # Count tokens per expert and build priority for dropping
    new_routing_weights = routing_weights.clone()

    for k in range(top_k):
        expert_ids = selected_experts[:, k]  # [n_tokens]

        for expert_idx in range(num_experts):
            mask = expert_ids == expert_idx
            token_indices = mask.nonzero(as_tuple=True)[0]

            if token_indices.numel() <= expert_capacity:
                continue

            # Drop excess tokens
            if drop_policy == "probs":
                probs_for_expert = router_probs[token_indices, expert_idx]
                _, sorted_idx = probs_for_expert.sort(descending=True)
                drop_indices = token_indices[sorted_idx[expert_capacity:]]
            else:
                drop_indices = token_indices[expert_capacity:]

            new_routing_weights[drop_indices, k] = 0.0

    return selected_experts, new_routing_weights


# === MoE Training Enhancer ===


class MoETrainingEnhancer:
    """Monkey-patches Lfm2MoeSparseMoeBlock.forward to inject training losses.

    After applying, each MoE block's forward will:
    1. Compute router logits and route tokens (original logic)
    2. Apply capacity-based token dropping (if configured)
    3. Compute aux loss + z-loss from router logits
    4. Inject losses via InjectAuxLoss.apply() on output
    5. Store metrics as module attributes for MoEMetricsCallback
    """

    def __init__(self, config: MoETrainingConfig):
        self.config = config

    def apply(self, model: nn.Module) -> None:
        """Patch all MoE blocks in the model."""
        patched = 0
        for module in model.modules():
            if type(module).__name__ == "Lfm2MoeSparseMoeBlock":
                self._patch_block(module)
                patched += 1
        logger.info(f"MoETrainingEnhancer: patched {patched} MoE blocks")

    def _patch_block(self, block: nn.Module) -> None:
        config = self.config
        original_forward = block.forward

        def enhanced_forward(hidden_states: torch.Tensor) -> torch.Tensor:
            B, S, H = hidden_states.shape
            x = hidden_states.view(-1, H)

            # === Router ===
            router_logits = block.gate(x)
            # Original routing — call the model's route_tokens_to_experts
            selected_experts, routing_weights = block.route_tokens_to_experts(
                router_logits
            )

            num_experts = router_logits.shape[-1]
            top_k = selected_experts.shape[-1]

            # Compute router probs for loss computation (sigmoid for LFM2 MoE)
            router_probs = torch.sigmoid(router_logits)

            # === Token dropping ===
            if config.capacity_factor is not None:
                selected_experts, routing_weights = apply_router_token_dropping(
                    router_probs=router_probs,
                    selected_experts=selected_experts,
                    routing_weights=routing_weights,
                    num_experts=num_experts,
                    capacity_factor=config.capacity_factor,
                    drop_policy=config.token_drop_policy,
                )

            # === Expert computation (use original forward logic) ===
            output = original_forward(hidden_states)

            # === Aux losses ===
            total_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

            if config.aux_loss_coef > 0:
                aux = switch_load_balancing_loss(
                    router_probs, selected_experts, num_experts, top_k, config.aux_loss_coef
                )
                total_aux_loss = total_aux_loss + aux
                block._moe_aux_loss = aux.detach()

            if config.z_loss_coef > 0:
                zl = z_loss(router_logits, config.z_loss_coef)
                total_aux_loss = total_aux_loss + zl
                block._moe_z_loss = zl.detach()

            # Inject loss into backward pass
            if total_aux_loss.item() != 0.0:
                output = InjectAuxLoss.apply(output, total_aux_loss)

            # === Store metrics for callback ===
            with torch.no_grad():
                tokens_per_expert = torch.zeros(
                    num_experts, device=x.device, dtype=torch.float32
                )
                ones = torch.ones_like(selected_experts[:, 0], dtype=torch.float32)
                for k in range(top_k):
                    tokens_per_expert.scatter_add_(0, selected_experts[:, k], ones)

                block._moe_tokens_per_expert = tokens_per_expert
                block._moe_router_logits_mean = router_logits.mean().item()
                block._moe_router_logits_std = router_logits.std().item()
                block._moe_num_experts = num_experts

            return output

        block.forward = enhanced_forward
