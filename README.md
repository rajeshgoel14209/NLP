pip install tabula-py pdf2image pytesseract opencv-python
pattern = r"\b(cover\s*page|executive\s*summary|sources?\s*of\s*repayment|financial\s*analysis|exposure\s*analysis|appendix)\b"
import cv2
import pytesseract
from pdf2image import convert_from_path
import tabula

# Convert image to PDF
def image_to_pdf(image_path, output_pdf):
    images = convert_from_path(image_path)
    images[0].save(output_pdf, "PDF")

# Extract tables using Tabula
def extract_tables_from_pdf(pdf_path):
    tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
    return tables

# Main process
image_path = "table_image.jpg"  # Replace with your image path
pdf_path = "output.pdf"

image_to_pdf(image_path, pdf_path)
tables = extract_tables_from_pdf(pdf_path)

# Print extracted tables
for idx, table in enumerate(tables):
    print(f"Table {idx+1}:\n", table)

pip install camelot-py[cv]

import camelot

# Convert Image to PDF first (same as previous example)

# Extract tables using Camelot
def extract_tables_with_camelot(pdf_path):
    tables = camelot.read_pdf(pdf_path, pages="1")
    return tables

# Extract tables
tables = extract_tables_with_camelot("output.pdf")

# Print extracted tables
for i, table in enumerate(tables):
    print(f"Table {i+1}:\n", table.df)


import cv2
import pytesseract
import pandas as pd

# Read the image
image = cv2.imread("table_image.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply OCR
custom_config = r"--oem 3 --psm 6"





Drawbacks of Mean Pooling for Chunked Embeddings
1. Loss of Important Information (Averaging Weakens Signal)
Mean pooling treats all words equally, so important words lose impact.
If a document contains a key idea in one chunk, averaging dilutes its influence.
2. Contextual Blurring (Mixing Multiple Meanings)
Different parts of a long document may have different meanings.
Averaging blends unrelated information into one representation.
3. Inability to Prioritize Important Chunks
Some chunks may be more relevant than others, but mean pooling gives equal weight to all chunks.



from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Example long text split into 3 chunks
chunks = [
    "Artificial intelligence is transforming industries by automating tasks.",
    "Self-driving cars use AI to navigate roads without human intervention.",
    "Deep learning models, like neural networks, are trained on massive datasets."
]

# Generate embeddings for each chunk
embeddings = model.encode(chunks)

# Display individual chunk embeddings (truncated)
for i, emb in enumerate(embeddings):
    print(f"Chunk {i+1} embedding (truncated):", emb[:5])  # Showing first 5 values only

# Apply Mean Pooling
mean_embedding = np.mean(embeddings, axis=0)

print("\nMean Pooled Embedding (truncated):", mean_embedding[:5])  # Truncated output


Chunk 1: [0.12, 0.34, 0.56, 0.78, 0.90, ...]
Chunk 2: [0.23, 0.45, 0.67, 0.89, 0.12, ...]
Chunk 3: [0.31, 0.51, 0.72, 0.93, 0.14, ...]

If Chunk 2 strongly represents "self-driving cars" and Chunk 3 represents "deep learning," the mean pooling result will blend these instead of keeping them distinct.

Before Mean Pooling
Chunk 1 (AI industry) → [0.12, 0.34, 0.56, 0.78, 0.90]
Chunk 2 (Self-driving cars) → [0.23, 0.45, 0.67, 0.89, 0.12]
Chunk 3 (Deep learning) → [0.31, 0.51, 0.72, 0.93, 0.14]

Mean Pooled: [0.22, 0.43, 0.65, 0.86, 0.38]

The sharp differences between "self-driving cars" and "deep learning" vanish.
The final embedding is less specific to any one concept.

Alternatives to Mean Pooling
✅ Max Pooling → Takes the strongest signals from each chunk.
✅ Weighted Mean Pooling → Gives more importance to key chunks.
✅ CLS Token Pooling → Uses the [CLS] token, which captures contextual meaning better.
✅ Hierarchical Embedding → Uses a separate transformer to combine chunk embeddings

from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Example long text split into 3 chunks
chunks = [
    "Artificial intelligence is transforming industries by automating tasks.",
    "Self-driving cars use AI to navigate roads without human intervention.",
    "Deep learning models, like neural networks, are trained on massive datasets."
]

# Generate embeddings for each chunk
embeddings = model.encode(chunks)

Step 3: Define Weights for Each Chunk
Weights can be assigned based on:
✅ Sentence Importance (e.g., TF-IDF score, attention score).
✅ Position in Text (e.g., intro & conclusion get higher weights).
✅ Predefined Importance Scores (e.g., domain knowledge).


# Example weights: First chunk is more important (higher weight)
weights = np.array([0.5, 0.3, 0.2])
weights = weights / np.sum(weights)  # Normalize to sum to 1

Step 4: Apply Weighted Pooling
Instead of a simple mean, multiply each embedding by its weight before summing.

python
Copy
Edit
# Apply weighted pooling
weighted_embedding = np.sum(embeddings * weights[:, np.newaxis], axis=0)

# Print first 5 values of the final pooled embedding
print("Weighted Pooled Embedding (truncated):", weighted_embedding[:5])



text = pytesseract.image_to_string(gray, config=custom_config)

# Print raw extracted text
print("Extracted Text:\n", text)





from sentence_transformers import SentenceTransformer
import numpy as np

# Load SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample texts
texts = [
    "This is a sample sentence.",
    "Another example text.",
    "AI is transforming the world.",
    "Machine learning is powerful."
]

# Generate embeddings using model.encode()
embeddings = model.encode(texts).astype('float32')




from langchain.vectorstores import FAISS
from langchain.schema import Document
import faiss
import pickle

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)  # Add embeddings to FAISS

# Store metadata separately (text content)
documents = [Document(page_content=text) for text in texts]

# Save FAISS index
faiss.write_index(index, "faiss_vectorstore.index")

# Save metadata (texts)
with open("faiss_metadata.pkl", "wb") as f:
    pickle.dump(documents, f)


    # Load FAISS index
index = faiss.read_index("faiss_vectorstore.index")

# Load metadata
with open("faiss_metadata.pkl", "rb") as f:
    documents = pickle.load(f)

# Create FAISS vector store using LangChain
vectorstore = FAISS(index, documents)


# Query text
query_text = "AI is changing technology."

# Generate query embedding
query_embedding = model.encode([query_text]).astype('float32')

# Perform search
results = vectorstore.search_by_vector(query_embedding[0], k=2)

# Print results
for doc in results:
    print("Similar text:", doc.page_content)
    
