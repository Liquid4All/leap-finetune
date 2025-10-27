"""
Data preprocessing utilities for Weave evaluation.

This module provides functions to prepare datasets for Weave evaluation,
including extracting messages and generating baseline outputs.
"""

import os


def get_messages_by_role(example, role: str) -> list[dict[str, str]]:
    """
    Extract messages for a specific role from a dataset example.

    Args:
        example: Dataset example with 'messages' field
        role: Role to filter by (e.g., 'user', 'assistant')

    Returns:
        List of message dictionaries for the specified role
    """
    return _get_messages_by_role(example["messages"], role=role)


def _get_messages_by_role(messages, role: str) -> list[dict[str, str]]:
    """
    Internal function to extract messages for a specific role.

    Args:
        messages: List of conversation messages
        role: Role to filter by

    Returns:
        List of message dictionaries for the specified role
    """
    return [conversation for conversation in messages if conversation["role"] == role]


def preprocess_datum(example, base_model):
    """
    Preprocess a single dataset example for Weave evaluation.

    Args:
        example: Dataset example with 'messages' field
        base_model: Base model instance with predict_sync method

    Returns:
        Dictionary with processed data for evaluation
    """
    messages_user = get_messages_by_role(example, role="user")
    messages_assistant = get_messages_by_role(example, role="assistant")

    # To simplify evaluation, let's make the dataset single-turn chat.
    # Note that we assume the dataset messages are properly formatted with
    # user and assistant messages interleaved at each turn, with the user message
    # preceding the assistant message.
    messages_user_first = messages_user[0]
    messages_assistant_first = messages_assistant[0]

    return {
        # Dataset must contain all inputs expected by the Model's `predict` method.
        "messages": [messages_user_first],
        # Fields needed for the `conversation_scorer` function.
        # Dataset rows must be a superset of fields expected by the Model's `predict`
        # method and any downstream scoring functions used in the Weave Evaluation.
        "reference_text": messages_assistant_first["content"],
        # You can compose an eval dataset in multiple novel ways
        "base_model_output": base_model.predict_sync([messages_user_first]),
    }


def preprocess_dataset_for_weave_evaluation_list(examples: list, base_model):
    """
    Preprocess a list of examples for Weave evaluation.

    Args:
        examples: List of dataset examples
        base_model: Base model instance with predict_sync method

    Returns:
        List of preprocessed examples
    """
    os.environ["WEAVE_PRINT_CALL_LINK"] = "false"
    preprocessed_data = [preprocess_datum(example, base_model) for example in examples]
    os.environ["WEAVE_PRINT_CALL_LINK"] = "true"
    return preprocessed_data


def preprocess_dataset_for_weave_evaluation(examples: dict, base_model):
    """
    Preprocess dataset examples in batched format for Weave evaluation.

    This function is designed to work with HuggingFace datasets' map() function
    with batched=True.

    Args:
        examples: Dictionary with batched dataset examples
        base_model: Base model instance with predict_sync method

    Returns:
        Dictionary with preprocessed batched data
    """
    os.environ["WEAVE_PRINT_CALL_LINK"] = "false"

    messages = examples["messages"]
    messages_user_firsts = [
        [_get_messages_by_role(conversation, role="user")[0]]
        for conversation in messages
    ]
    reference_texts = [
        _get_messages_by_role(conversation, role="assistant")[0]["content"]
        for conversation in messages
    ]
    base_model_outputs = [
        base_model.predict_sync(messages) for messages in messages_user_firsts
    ]

    os.environ["WEAVE_PRINT_CALL_LINK"] = "true"

    return {
        # Dataset must contain all inputs expected by the Model's `predict` method.
        "messages": messages_user_firsts,
        # Fields needed for the `conversation_scorer` function.
        # Dataset rows must be a superset of fields expected by the Model's `predict`
        # method and any downstream scoring functions used in the Weave Evaluation.
        "reference_text": reference_texts,
        # You can compose an eval dataset in multiple novel ways
        "base_model_output": base_model_outputs,
    }
