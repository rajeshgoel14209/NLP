https://ai.plainenglish.io/table-extraction-using-llms-unlocking-structured-data-from-documents-50cf21c98509

https://towardsdatascience.com/5-proven-query-translation-techniques-to-boost-your-rag-performance-47db12efe971

https://blog.stackademic.com/late-chunking-embedding-first-chunk-later-long-context-retrieval-in-rag-applications-3a292f6443bb

import numpy as np

def weighted_mean_pooling(embeddings: list, weighting_strategy="equal"):
    """
    Apply weighted mean pooling on a variable-length list of embeddings.

    :param embeddings: List of numpy arrays (each embedding must have the same dimension)
    :param weighting_strategy: Strategy to generate weights dynamically
    :return: Weighted mean pooled embedding (numpy array)
    """
    num_embeddings = len(embeddings)
    embeddings = np.array(embeddings)  # Convert list to numpy array

    # Generate dynamic weights
    if weighting_strategy == "equal":
        weights = np.ones(num_embeddings) / num_embeddings  # Equal distribution
    elif weighting_strategy == "linear_decay":
        weights = np.linspace(1, 0.5, num_embeddings)  # Linearly decreasing weights
    elif weighting_strategy == "exponential_decay":
        weights = np.exp(-np.arange(num_embeddings))  # Exponentially decreasing weights
    else:
        raise ValueError("Unknown weighting strategy")

    weights = weights.reshape(-1, 1)  # Reshape for broadcasting

    # Compute weighted sum
    weighted_sum = np.sum(embeddings * weights, axis=0)

    # Normalize by total weight
    weighted_mean = weighted_sum / np.sum(weights)

    return weighted_mean

# Example: Variable number of embeddings
embeddings_list = [
    np.array([0.1, 0.3, 0.5, 0.7, 0.2]),
    np.array([0.4, 0.2, 0.6, 0.5, 0.9]),
    np.array([0.2, 0.7, 0.3, 0.9, 0.4])
]

# Apply weighted mean pooling with equal weights
pooled_embedding = weighted_mean_pooling(embeddings_list, weighting_strategy="equal")
print("Pooled Embedding:", pooled_embedding)

