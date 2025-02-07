# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="my_collection")

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Data to add
documents = ["This is document one.", "This is document two."]
metadata_list = [{"source": "doc1", "category": "news"}, {"source": "doc2", "category": "sports"}]

# Generate embeddings
embeddings = [model.encode(text).tolist() for text in documents]

# Add to ChromaDB
collection.add(
    ids=["1", "2"],
    embeddings=embeddings,
    documents=documents,
    metadatas=metadata_list
)












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

    def add_documents(self, documents, fresh_load=False):
        if not isinstance(documents, list) or not all(isinstance(d, dict) for d in documents):
            logging.error("Invalid documents format. Expected a list of dictionaries.")
            return False
        
        if fresh_load:
            try:
                self.collection.delete()
                logging.info("Fresh load: Existing collection cleared.")
            except Exception as e:
                logging.error(f"Failed to clear collection: {e}")
                return False
        
        success = True
        for doc in documents:
            doc_id = doc.get("id")
            text = doc.get("text")
            metadata = doc.get("metadata", {})
            
            if not doc_id or not text:
                logging.error("Document ID and text are required.")
                success = False
                continue
            
            embedding = self.generate_embedding(text)
            if embedding is None:
                logging.error(f"Skipping document {doc_id} due to embedding failure.")
                success = False
                continue
            
            try:
                self.collection.add(ids=[doc_id], embeddings=[embedding], documents=[text], metadatas=[metadata])
                logging.info(f"Document {doc_id} added successfully.")
            except ChromaError as e:
                logging.error(f"ChromaDB error while adding document {doc_id}: {e}")
                success = False
            except Exception as e:
                logging.error(f"Unexpected error while adding document {doc_id}: {e}")
                success = False
        return success

    def query(self, text, n_results=5, filter_metadata=None):
        embedding = self.generate_embedding(text)
        if embedding is None:
            logging.error("Query failed due to embedding generation error.")
            return []
        
        try:
            results = self.collection.query(query_embeddings=[embedding], n_results=n_results, where=filter_metadata or {})
            return results
        except ChromaError as e:
            logging.error(f"ChromaDB query error: {e}")
        except Exception as e:
            logging.error(f"Unexpected error during query: {e}")
        return []

# Example usage
if __name__ == "__main__":
    chroma_handler = ChromaDBHandler()
    documents = [
        {"id": "1", "text": "This is a sample document.", "metadata": {"category": "example"}},
        {"id": "2", "text": "Another document with a different category.", "metadata": {"category": "test"}}
    ]
    chroma_handler.add_documents(documents, fresh_load=True)
    response = chroma_handler.query("sample", filter_metadata={"category": "example"})
    print("Query results:", response)
