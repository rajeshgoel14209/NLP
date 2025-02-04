https://ai.plainenglish.io/table-extraction-using-llms-unlocking-structured-data-from-documents-50cf21c98509

https://towardsdatascience.com/5-proven-query-translation-techniques-to-boost-your-rag-performance-47db12efe971

https://blog.stackademic.com/late-chunking-embedding-first-chunk-later-long-context-retrieval-in-rag-applications-3a292f6443bb

import faiss
import numpy as np
from langchain.vectorstores import FAISS
from langchain.schema import Document

# Sample custom embeddings (5D for demo, real ones are 768D+)
custom_embeddings = np.array([
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.5, 0.4, 0.3, 0.2, 0.1],
    [0.9, 0.8, 0.7, 0.6, 0.5]
], dtype=np.float32)

# Associated texts & metadata
documents = [
    Document(page_content="This is document 1", metadata={"category": "science"}),
    Document(page_content="This is document 2", metadata={"category": "math"}),
    Document(page_content="This is document 3", metadata={"category": "history"}),
]

# Create FAISS index
dimension = custom_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)

# Add embeddings to FAISS
faiss_index.add(custom_embeddings)

# Wrap FAISS in LangChain's FAISS VectorStore
vectorstore = FAISS(faiss_index, documents)

# Save FAISS index
vectorstore.save_local("faiss_store")


# Load FAISS vector store
vectorstore = FAISS.load_local("faiss_store", allow_dangerous_deserialization=True)


