import faiss
import numpy as np

# Step 1: Create a FAISS index and add some embeddings
dimension = 128  # Example embedding size
index = faiss.IndexFlatL2(dimension)  # L2 distance-based index

# Add some random embeddings (replace with your actual embeddings)
num_embeddings = 10
embeddings = np.random.rand(num_embeddings, dimension).astype('float32')
index.add(embeddings)

# Step 2: Define the indices you want to filter
indices_to_filter = [2, 5, 7]

# Step 3: Retrieve embeddings for the given indices
filtered_embeddings = []
for idx in indices_to_filter:
    embedding = index.reconstruct(idx)  # Retrieve embedding for the given index
    filtered_embeddings.append(embedding)

# Convert to a NumPy array for further use
filtered_embeddings = np.array(filtered_embeddings)

# Print results
print("Filtered Embeddings Shape:", filtered_embeddings.shape)
print("Filtered Embeddings:", filtered_embeddings)
