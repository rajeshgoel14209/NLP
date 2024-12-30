import faiss
import numpy as np

# Example metadata and embeddings
metadata = [
    {"id": 1, "name": "Alice", "age": 30, "occupation": "Engineer"},
    {"id": 2, "name": "Bob", "age": 25, "occupation": "Designer"},
    {"id": 3, "name": "Charlie", "age": 35, "occupation": "Teacher"},
]

# Example embeddings (3D vectors for demonstration)
embeddings = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9]
], dtype=np.float32)

# Build FAISS index
dimension = embeddings.shape[1]  # Dimensionality of the vectors
index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean) distance
index.add(embeddings)  # Add embeddings to the index

# Filter function for metadata
def filter_metadata(metadata, condition):
    """
    Filters metadata based on a condition.

    Args:
        metadata (list): List of metadata dictionaries.
        condition (callable): A function that returns True for desired metadata.

    Returns:
        list: Indices of metadata items that meet the condition.
    """
    return [i for i, meta in enumerate(metadata) if condition(meta)]

# Function for FAISS search with metadata filtering
def search_with_metadata(query_vector, index, metadata, condition, top_k=2):
    """
    Perform FAISS search with filtering based on metadata.

    Args:
        query_vector (np.ndarray): Query vector (1D array).
        index (faiss.Index): FAISS index.
        metadata (list): List of metadata dictionaries.
        condition (callable): A function to filter metadata.
        top_k (int): Number of top results to retrieve.

    Returns:
        list: Filtered search results with metadata and distances.
    """
    # Filter metadata and get corresponding indices
    filtered_indices = filter_metadata(metadata, condition)
    
    # Subset embeddings based on filtered indices
    filtered_embeddings = np.array([embeddings[i] for i in filtered_indices])
    if filtered_embeddings.shape[0] == 0:
        return []  # No matches

    # Create a temporary FAISS index for the filtered embeddings
    temp_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
    temp_index.add(filtered_embeddings)

    # Perform search
    query_vector = query_vector.reshape(1, -1).astype(np.float32)
    distances, indices = temp_index.search(query_vector, top_k)

    # Map indices back to original metadata
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:  # Ensure valid index
            original_idx = filtered_indices[idx]
            result = metadata[original_idx]
            result["distance"] = dist
            results.append(result)
    return results

# Example query vector
query = np.array([0.1, 0.2, 0.25], dtype=np.float32)

# Example condition: Filter for "Engineers" only
condition = lambda meta: meta["occupation"] == "Engineer"

# Perform search with metadata filtering
results = search_with_metadata(query, index, metadata, condition, top_k=2)

# Print results
print("Filtered Search Results:")
for result in results:
    print(result)
