def is_moe_model_from_name(model_name: str) -> bool:
    moe_indicators = ["8B-A1B", "8BA1B", "24B-A2B", "24BA2B", "moe", "MoE"]
    return any(indicator.lower() in model_name.lower() for indicator in moe_indicators)


def is_large_moe_model_from_name(model_name: str) -> bool:
    """Check if model is a large MoE (24B+) that needs full_shard FSDP."""
    large_moe_indicators = ["24B-A2B", "24BA2B"]
    return any(indicator.lower() in model_name.lower() for indicator in large_moe_indicators)
