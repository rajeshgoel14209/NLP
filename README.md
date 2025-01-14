import faiss
import numpy as np

# Step 1: Create an IndexFlatL2 and wrap it with IndexIDMap
dimension = 128  # Size of the embeddings
base_index = faiss.IndexFlatL2(dimension)  # Base index
index = faiss.IndexIDMap(base_index)  # Wrapper to support custom IDs

# Step 2: Add embeddings with custom IDs
num_embeddings = 5
embeddings = np.random.rand(num_embeddings, dimension).astype('float32')  # Random embeddings
custom_ids = np.array([101, 102, 103, 200, 300]).astype('int64')  # Custom IDs

index.add_with_ids(embeddings, custom_ids)

# Verify that embeddings and IDs are added
print(f"Number of embeddings in the index: {index.ntotal}")
print(f"Stored IDs: {list(index.id_map)}")

# Step 3: Retrieve an embedding by custom ID
def get_embedding_by_id(index, custom_id):
    """
    Retrieve an embedding for a given custom ID from the FAISS index.
    """
    if not isinstance(index, faiss.IndexIDMap):
        raise ValueError("Index does not support ID-based retrieval. Wrap it with IndexIDMap.")

    # Convert Int64Vector to list for easier ID mapping
    id_list = list(index.id_map)

    try:
        # Find the internal index corresponding to the custom ID
        internal_idx = id_list.index(custom_id)

        # Retrieve the embedding using the internal index
        embedding = index.reconstruct(internal_idx)
        return embedding
    except ValueError:
        raise ValueError(f"Custom ID {custom_id} not found in the FAISS index.")

# Retrieve embedding for a specific ID
try:
    custom_id = 102
    embedding = get_embedding_by_id(index, custom_id)
    print(f"Embedding for ID {custom_id} retrieved successfully. Shape: {embedding.shape}")
except ValueError as e:
    print(e)
