from transformers import AutoTokenizer, AutoModel

def check_max_token_capacity_gte(model_name):
    """
    Check the maximum token capacity of a GTE (or similar transformer-based) model.
    
    Args:
        model_name (str): Name of the model (e.g., 'gte-base', 'gte-large').
    
    Returns:
        dict: A dictionary with model details including max token capacity.
    """
    try:
        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Get the maximum token capacity
        max_token_capacity = model.config.max_position_embeddings

        return {
            "model_name": model_name,
            "max_token_capacity": max_token_capacity
        }
    except Exception as e:
        return {"error": str(e)}

# Example Usage
model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Replace with GTE model name
result = check_max_token_capacity_gte(model_name)

# Display the result
if "error" in result:
    print(f"Error: {result['error']}")
else:
    print(f"Model Name: {result['model_name']}")
    print(f"Max Token Capacity: {result['max_token_capacity']}")
