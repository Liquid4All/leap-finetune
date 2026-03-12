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
