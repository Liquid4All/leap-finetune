# === HF FSDP defaults ===
# These are the baseline FSDP1 shapes used by config tests and older HF paths.
# The optimized MoE runners resolve their runtime FSDP2 flags below.

MOE_FSDP_CONFIG = {
    "fsdp": ["shard_grad_op", "auto_wrap"],
    "fsdp_config": {
        "transformer_layer_cls_to_wrap": "Lfm2MoeDecoderLayer",
        "backward_prefetch": "backward_pre",
        "sync_module_states": True,
        "use_orig_params": True,
    },
}

MOE_FSDP_CONFIG_LARGE = {
    "fsdp": ["full_shard", "auto_wrap"],
    "fsdp_config": {
        "transformer_layer_cls_to_wrap": "Lfm2MoeDecoderLayer",
        "backward_prefetch": "backward_pre",
        "sync_module_states": True,
        "use_orig_params": True,
        "activation_checkpointing": True,
    },
}


def resolve_reshard_after_forward(train_config: dict, default: bool) -> bool:
    """Prefer explicit YAML reshard policy, otherwise use the runner default."""
    config_value = train_config.get("reshard_after_forward")
    if config_value is not None:
        return bool(config_value)
    return default


def resolve_fsdp_cpu_offload(train_config: dict, default: bool = False) -> bool:
    """Prefer explicit YAML CPU offload policy, otherwise use the runner default."""
    config_value = train_config.get("fsdp_cpu_offload")
    if config_value is not None:
        return bool(config_value)
    return default
