import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import cityblock, minkowski

def calculate_similarity_or_distance(embeddings, query_vector, metric="cosine", top_k=10):
    """
    Wrapper function to calculate similarity or distance between vector embeddings and a query vector.

    Args:
        embeddings (np.ndarray): Array of shape (n_samples, n_features) containing embeddings.
        query_vector (np.ndarray): Array of shape (n_features,) representing the query embedding.
        metric (str): Metric to use. Options are "cosine", "euclidean", "manhattan", "minkowski".
        top_k (int): Number of closest embeddings to return.

    Returns:
        list: Top K indices and their scores based on the chosen metric.
    """
    # Ensure query_vector is 2D
    query_vector = query_vector.reshape(1, -1)

    # Calculate similarity or distance
    if metric == "cosine":
        scores = cosine_similarity(embeddings, query_vector).flatten()
        top_k_indices = np.argsort(scores)[::-1][:top_k]  # Descending order
    elif metric == "euclidean":
        scores = euclidean_distances(embeddings, query_vector).flatten()
        top_k_indices = np.argsort(scores)[:top_k]  # Ascending order
    elif metric == "manhattan":
        scores = np.array([cityblock(embed, query_vector.flatten()) for embed in embeddings])
        top_k_indices = np.argsort(scores)[:top_k]  # Ascending order
    elif metric == "minkowski":
        scores = np.array([minkowski(embed, query_vector.flatten(), p=3) for embed in embeddings])
        top_k_indices = np.argsort(scores)[:top_k]  # Ascending order
    else:
        raise ValueError(f"Unsupported metric: {metric}. Choose from 'cosine', 'euclidean', 'manhattan', or 'minkowski'.")

    # Return top K indices and their scores
    return [(idx, scores[idx]) for idx in top_k_indices]

# Example usage
if __name__ == "__main__":
    np.random.seed(42)

    # Generate random embeddings (n_samples=100, n_features=512)
    embeddings = np.random.rand(100, 512)
