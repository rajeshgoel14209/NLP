from transformers import GPT2Tokenizer, GPT2Model
import numpy as np

def check_max_token_capacity_and_create_embedding(text, model_name='gpt2'):
    """
    Check the maximum token length for a model, create embeddings, 
    and revert embeddings back to the original text.
    
    Args:
        text (str): The input text to process.
        model_name (str): Name of the transformer model (e.g., 'gpt2').
    
    Returns:
        dict: A dictionary with token length, embeddings, and reverted text.
    """
    # Load the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2Model.from_pretrained(model_name)
    
    # Tokenize the input text
    tokens = tokenizer.encode(text, return_tensors='pt')
    
    # Get the token length
    token_length = tokens.size(1)
    
    # Check maximum token capacity of the model
    max_token_capacity = model.config.n_positions
    
    if token_length > max_token_capacity:
        raise ValueError(f"Input exceeds the max token length of {max_token_capacity} tokens.")

    # Generate embeddings
    with torch.no_grad():
        outputs = model(tokens)
        embeddings = outputs.last_hidden_state.numpy()  # Convert to numpy for inspection
    
    # Decode tokens back to original text
    reverted_text = tokenizer.decode(tokens[0], skip_special_tokens=True)
    
    return {
        "token_length": token_length,
        "max_token_capacity": max_token_capacity,
        "embeddings": embeddings,
        "reverted_text": reverted_text
    }


# Example Usage
text = "This is a test input to check the token length, generate embeddings, and revert back to text."
result = check_max_token_capacity_and_create_embedding(text)

# Display Results
print(f"Token Length: {result['token_length']}")
print(f"Max Token Capacity: {result['max_token_capacity']}")
print(f"Reverted Text: {result['reverted_text']}")
print(f"Embeddings Shape: {result['embeddings'].shape}")
