import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.utils.errors import ChromaError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChromaDBHandler:
    def __init__(self, collection_name="default_collection", embedding_model="all-MiniLM-L6-v2"):
        try:
            self.client = chromadb.PersistentClient(path="./chroma_db")  # Persistent storage
            self.collection = self.client.get_or_create_collection(collection_name)
            self.model = SentenceTransformer(embedding_model)
            logging.info("ChromaDB and embedding model initialized successfully.")
        except Exception as e:
            logging.error(f"Initialization failed: {e}")
            raise

    def generate_embedding(self, text):
        if not text or not isinstance(text, str):
            logging.error("Invalid text input for embedding generation.")
            return None
        try:
            return self.model.encode(text, convert_to_numpy=True).tolist()
        except Exception as e:
            logging.error(f"Embedding generation failed: {e}")
            return None

    def add_document(self, doc_id, text, metadata=None):
        if not doc_id or not text:
            logging.error("Document ID and text are required.")
            return False
        
        embedding = self.generate_embedding(text)
        if embedding is None:
            logging.error("Skipping document insertion due to embedding failure.")
            return False
        
        try:
            self.collection.add(ids=[doc_id], embeddings=[embedding], documents=[text], metadatas=[metadata or {}])
            logging.info(f"Document {doc_id} added successfully.")
            return True
        except ChromaError as e:
            logging.error(f"ChromaDB error while adding document: {e}")
        except Exception as e:
            logging.error(f"Unexpected error while adding document: {e}")
        return False

    def query(self, text, n_results=5):
        embedding = self.generate_embedding(text)
        if embedding is None:
            logging.error("Query failed due to embedding generation error.")
            return []
        
        try:
            results = self.collection.query(query_embeddings=[embedding], n_results=n_results)
            return results
        except ChromaError as e:
            logging.error(f"ChromaDB query error: {e}")
        except Exception as e:
            logging.error(f"Unexpected error during query: {e}")
        return []

# Example usage
if __name__ == "__main__":
    chroma_handler = ChromaDBHandler()
    chroma_handler.add_document("1", "This is a sample document.")
    response = chroma_handler.query("sample")
    print("Query results:", response)
