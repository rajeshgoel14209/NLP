import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_embeddings(embeddings, query_vector, labels=None, title="Embedding Distribution"):
    """
    Visualize vector embeddings and query vector in 2D space.
    
    Args:
        embeddings (np.ndarray): Array of shape (n_samples, n_features) containing embeddings.
        query_vector (np.ndarray): Array of shape (n_features,) representing the query embedding.
        labels (list): Optional list of labels for each embedding (e.g., metadata or cluster).
        title (str): Title for the plot.
    """
    # Ensure query_vector is 2D for concatenation
    query_vector = query_vector.reshape(1, -1)
    
    # Combine embeddings and query vector
    all_vectors = np.vstack([embeddings, query_vector])
    
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(all_vectors)
    
    # Split embeddings and query for visualization
    reduced_embeddings = reduced_vectors[:-1]
    reduced_query = reduced_vectors[-1]
    
    # Plot the embeddings
    plt.figure(figsize=(10, 7))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c='blue', label="Embeddings", alpha=0.7)
    
    # Highlight the query vector
    plt.scatter(reduced_query[0], reduced_query[1], c='red', label="Query Vector", marker='X', s=150)
    
    # Optionally annotate with labels
    if labels:
        for i, label in enumerate(labels):
            plt.annotate(label, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=8, alpha=0.75)
    
    # Add title and legend
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# Example usage
# Generate some random embeddings (n_samples=100, n_features=512)
np.random.seed(42)
embeddings = np.random.rand(100, 512)

# Create a random query vector
query_vector = np.random.rand(512)

# Example metadata labels
labels = [f"Vec {i}" for i in range(len(embeddings))]

# Visualize
visualize_embeddings(embeddings, query_vector, labels=labels)
Explanation
Dimensionality Reduction:

High-dimensional embeddings (e.g., 512 dimensions) are reduced to 2D using PCA for visualization.
You can also use t-SNE or UMAP for a non-linear reduction.
Query Vector:

The query vector is highlighted separately in red with a larger marker.
Optional Labels:

Metadata or labels for embeddings can be displayed for interpretability.
Distribution:

The scatter plot shows how the embeddings are distributed relative to the query vector.
Optional: Use t-SNE for Better Clustering
python
Copy
Edit
from sklearn.manifold import TSNE

# Use t-SNE instead of PCA
tsne = TSNE(n_components=2, random_state=42)
reduced_vectors_tsne = tsne.fit_transform(np.vstack([embeddings, query_vector]))

# Split embeddings and query vector
reduced_embeddings_tsne = reduced_vectors_tsne[:-1]
reduced_query_tsne = reduced_vectors_tsne[-1]

# Plot (similar to the PCA example)
This approach provides an intuitive way to compare the query vector's position relative to the embedding distribution!







