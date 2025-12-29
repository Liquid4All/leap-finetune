def is_moe_model_from_name(model_name: str) -> bool:
    """Check if a model is a MoE (Mixture of Experts) model based on model name.

    Args:
        model_name: The model name/identifier

    Returns:
        True if the model is a MoE model, False otherwise
    """
    # Check for MoE model identifiers in the name
    moe_indicators = ["8B-A1B", "8BA1B", "moe", "MoE"]
    return any(indicator.lower() in model_name.lower() for indicator in moe_indicators)
