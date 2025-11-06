"""
Utility functions for LLM Router
"""

def get_short_model_prefix(model_id: str) -> str:
    """
    Get a short prefix for the model ID for file naming
    
    Args:
        model_id: Full model ID
    
    Returns:
        Short model prefix
    """
    # Handle OpenAI models
    if model_id.startswith("gpt-"):
        return model_id.replace(".", "_")
    
    # Handle Claude models
    if "claude" in model_id:
        parts = model_id.split("-")
        return f"claude_{parts[1]}_{parts[2]}" if len(parts) >= 3 else model_id.replace("-", "_")
    
    # Handle Gemini models
    if model_id.startswith("gemini-"):
        return model_id.replace(".", "_").replace("-", "_")
    
    # Handle Together AI / DeepSeek / other full path models
    if "/" in model_id:
        # Extract just the model name after the slash
        short_name = model_id.split("/")[-1]
        # Simplify common patterns
        short_name = short_name.replace("Instruct", "").replace("Turbo", "")
        short_name = short_name.replace("-", "_").replace(".", "_")
        return short_name.lower()
    
    # Handle XAI models
    if "grok" in model_id:
        return model_id.replace("-", "_")
    
    # Default: replace special characters
    return model_id.replace("/", "_").replace("-", "_").replace(".", "_").lower()
