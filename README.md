pip install faiss-cpu sentence-transformers torchvision Pillow

import torch
from torchvision import models, transforms
from PIL import Image
import faiss
import numpy as np

# Step 1: Load a Pretrained Image Embedding Model
model = models.resnet50(pretrained=True)  # Using ResNet50 as an example
model.eval()  # Set the model to evaluation mode

# Step 2: Preprocessing Transformation for Images
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Step 3: Function to Generate Image Embeddings
def generate_embedding(image_path):
    image = Image.open(image_path).convert("RGB")  # Load image
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        embedding = model(input_tensor).squeeze().numpy()  # Extract embedding
    return embedding

# Example Image Paths
image_paths = ["image1.jpg", "image2.jpg"]

# Step 4: Generate Embeddings for All Images
embeddings = [generate_embedding(img_path) for img_path in image_paths]

# Step 5: Create FAISS Index
dimension = embeddings[0].shape[0]  # Dimensionality of the embeddings
faiss_index = faiss.IndexFlatL2(dimension)  # L2 distance metric

# Step 6: Add Embeddings to the FAISS Index
embeddings_array = np.array(embeddings).astype("float32")
faiss_index.add(embeddings_array)

# Step 7: Save Metadata Separately (if needed)
metadata = [{"image_path": img_path} for img_path in image_paths]

# Save the FAISS index
faiss.write_index(faiss_index, "image_index.faiss")

print("Image embeddings stored in FAISS index.")


python
Copy code
# Load FAISS Index
faiss_index = faiss.read_index("image_index.faiss")

# Query with an Example Image
query_embedding = generate_embedding("query_image.jpg").astype("float32")
k = 3  # Number of similar results to retrieve

# Perform Similarity Search
distances, indices = faiss_index.search(np.array([query_embedding]), k)

# Retrieve Results
for idx, distance in zip(indices[0], distances[0]):
    if idx != -1:  # Ensure valid result
        print(f"Matched Image: {metadata[idx]['image_path']}, Distance: {distance}")


from sentence_transformers import SentenceTransformer
import numpy as np

# Load CLIP Model
model = SentenceTransformer('clip-ViT-B-32')

# Generate Image Embedding
image_embedding = model.encode(Image.open("image.jpg"), convert_to_tensor=True)

# Add to FAISS Index
faiss_index.add(image_embedding.numpy())


