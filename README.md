import faiss
import numpy as np

# Step 1: Create an IndexFlatL2 index
dimension = 128  # Embedding size
index = faiss.IndexFlatL2(dimension)  # L2 distance-based index

# Step 2: Add data with custom IDs
num_embeddings = 10
embeddings = np.random.rand(num_embeddings, dimension).astype('float32')

# Define custom IDs (non-sequential)
custom_ids = np.array([101, 102, 103, 200, 300, 400, 500, 600, 700, 800]).astype('int64')
index.add_with_ids(embeddings, custom_ids)

# Step 3: Retrieve embeddings based on custom IDs
def get_embeddings_by_ids(index, ids_to_filter):
    """
    Retrieve embeddings for the given custom IDs.
    """
    retrieved_embeddings = []
    for custom_id in ids_to_filter:
        try:
            # Map custom ID to internal FAISS index
            internal_index = np.where(custom_ids == custom_id)[0]
            if len(internal_index) == 0:
                raise ValueError(f"ID {custom_id} not found in FAISS index.")
            
            internal_index = internal_index[0]
            
            # Retrieve embedding
            embedding = index.reconstruct(internal_index)
            retrieved_embeddings.append({"id": custom_id, "embedding": embedding})
        except Exception as e:
            print(f"Error retrieving embedding for ID {custom_id}: {e}")
    
    return retrieved_embeddings

# Step 4: Define custom IDs to filter
ids_to_filter = [102, 300, 700, 999]  # 999 does not exist for testing error handling

# Retrieve the embeddings
filtered_embeddings = get_embeddings_by_ids(index, ids_to_filter)

# Display results
for result in filtered_embeddings:
    print(f"ID: {result['id']}, Embedding Shape: {result['embedding'].shape}")
