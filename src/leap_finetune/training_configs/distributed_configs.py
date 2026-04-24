import copy

########################
#     FSDP CONFIGS     #
########################

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


def resolve_reshard_after_forward(
    train_config: dict, default: bool
) -> bool:
    """Resolve reshard_after_forward from config, then default."""
    config_value = train_config.get("reshard_after_forward")
    if config_value is not None:
        return bool(config_value)
    return default


def resolve_fsdp_cpu_offload(
    train_config: dict, default: bool = False
) -> bool:
    """Resolve fsdp_cpu_offload from config, then default."""
    config_value = train_config.get("fsdp_cpu_offload")
    if config_value is not None:
        return bool(config_value)
    return default


def build_moe_fsdp_config(
    *, reshard_after_forward: bool, activation_checkpointing: bool = False
) -> dict:
    """Build an HF FSDP config that mirrors the requested reshard mode."""
    template = MOE_FSDP_CONFIG_LARGE if reshard_after_forward else MOE_FSDP_CONFIG
    config = copy.deepcopy(template)
    fsdp_config = config["fsdp_config"]
    if activation_checkpointing:
        fsdp_config["activation_checkpointing"] = True
    else:
        fsdp_config.pop("activation_checkpointing", None)
    return config
