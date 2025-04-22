import json
from typing import Dict, Any


def calculate_cost(prompt_tokens, completion_tokens, prompt_price, completion_price):
    """
    Calculate the cost in cents based on token usage and provider

    Args:
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        prompt_price: Price per million tokens for prompts
        completion_price: Price per million tokens for completions

    Returns:
        Cost in cents or None if pricing info not available
    """
    # Check if pricing info is available
    if prompt_price is None or completion_price is None:
        return None

    # Calculate costs based on established pricing
    prompt_cost = prompt_tokens * (prompt_price / 1_000_000)
    completion_cost = completion_tokens * (completion_price / 1_000_000)

    # Return total cost in cents
    return prompt_cost + completion_cost


def extract_token_usage(response) -> Dict[str, int]:
    """
    Extract token usage information from the LLM response

    Args:
        response: The LLM response object

    Returns:
        Dictionary with token counts
    """
    # Initialize token counts
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    token_usage = None

    # Extract token usage from the response
    if hasattr(response, "usage") and response.usage:
        token_usage = response.usage
    elif hasattr(response, "response_metadata") and response.response_metadata:
        if "usage" in response.response_metadata:
            token_usage = response.response_metadata["usage"]
        elif "token_usage" in response.response_metadata:
            token_usage = response.response_metadata["token_usage"]

    # Extract token counts from token_usage
    if token_usage:
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        total_tokens = token_usage.get("total_tokens", 0)
        # Calculate total if not provided
        if total_tokens == 0 and (prompt_tokens > 0 or completion_tokens > 0):
            total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
