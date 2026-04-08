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


def get_model_family(model_name: str) -> str:
    """Return model family for format-specific behavior.

    Returns 'lfm25' for LFM2.5 models, 'lfm2' for LFM2 models.
    LFM2-24B-A2B is the only exception — uses LFM2.5 format despite the LFM2 name.
    """
    if "2.5" in model_name:
        return "lfm25"
    # LFM2-24B-A2B is the only LFM2 model that uses LFM2.5 format
    if "24B" in model_name and "2.5" not in model_name:
        return "lfm25"
    return "lfm2"
