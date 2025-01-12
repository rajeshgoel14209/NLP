def validate_data(data):
    """
    Validates a list of dictionaries to ensure they only contain the keys 'metadata' and 'chunk_text'.
    Any dictionary with more than these keys is ignored.
    If either 'metadata' or 'chunk_text' is missing in any dictionary, raise a ValueError.
    
    Args:
        data (list): List of dictionaries to validate.
    
    Returns:
        list: A filtered list of dictionaries with only valid entries.
    
    Raises:
        ValueError: If any dictionary is missing the required keys.
    """
    required_keys = {"metadata", "chunk_text"}
    validated_data = []

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {i} is not a dictionary.")

        # Check if all required keys are present
        if not required_keys.issubset(item.keys()):
            missing_keys = required_keys - item.keys()
            raise ValueError(f"Item at index {i} is missing required keys: {missing_keys}")

        # Ignore dictionaries with extra keys
        if len(item.keys()) > len(required_keys):
            continue

        # Add valid item to the filtered list
        validated_data.append({key: item[key] for key in required_keys})

    return validated_data
	
	
try:
    valid_data = validate_data(data)
    print("Validated Data:", valid_data)
except ValueError as e:
    print("Validation Error:", e)
	
	
	
	
def validate_chunk_text(data):
    """
    Validates that the 'chunk_text' in each dictionary of a list is a string.
    
    Args:
        data (list): List of dictionaries to validate.
    
    Returns:
        bool: True if all validations pass.
    
    Raises:
        ValueError: If any 'chunk_text' is not a string.
    """
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {i} is not a dictionary.")
        
        if 'chunk_text' not in item:
            raise ValueError(f"'chunk_text' is missing in dictionary at index {i}.")
        
        if not isinstance(item['chunk_text'], str):
            raise ValueError(f"'chunk_text' at index {i} is not a string. Found: {type(item['chunk_text']).__name__}")
    
    return True




try:
    validate_chunk_text(data)
    print("All chunk_text values are valid strings.")
except ValueError as e:
    print("Validation Error:", e)



	
from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
import os

def store_in_vector_db_langchain(data, db_type="faiss", faiss_index_path="faiss_index", chroma_db_path="chroma_storage"):
    """
    Stores data in a vector database (FAISS or ChromaDB) using LangChain.

    Args:
        data (list): List of dictionaries with keys `metadata` and `chunk_text`.
        db_type (str): Type of vector database to use ('faiss' or 'chroma').
        faiss_index_path (str): Path to save/load the FAISS index.
        chroma_db_path (str): Path to store ChromaDB data.

    Returns:
        None

    Raises:
        ValueError: If input data is invalid or db_type is unsupported.
    """
    # Validate input data
    if not isinstance(data, list):
        raise ValueError("Input data must be a list of dictionaries.")
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {i} is not a dictionary.")
        if "metadata" not in item or "chunk_text" not in item:
            raise ValueError(f"Item at index {i} must have 'metadata' and 'chunk_text' keys.")
        if not isinstance(item["chunk_text"], str):
            raise ValueError(f"'chunk_text' at index {i} must be a string.")

    # Convert data to LangChain Document format
    documents = [
        Document(page_content=item["chunk_text"], metadata=item["metadata"])
        for item in data
    ]

    # Initialize embeddings
    embeddings = OpenAIEmbeddings()  # Replace with your preferred embedding model

    if db_type == "faiss":
        # Handle FAISS vector database
        try:
            if os.path.exists(faiss_index_path):
                # Load existing FAISS index
                vector_db = FAISS.load_local(faiss_index_path, embeddings)
            else:
                # Create new FAISS index
                vector_db = FAISS.from_documents(documents, embeddings)

            # Save updated FAISS index
            vector_db.save_local(faiss_index_path)
            print(f"Data stored in FAISS index at {faiss_index_path}.")
        except Exception as e:
            raise RuntimeError(f"Error handling FAISS database: {e}")

    elif db_type == "chroma":
        # Handle Chroma vector database
        try:
            vector_db = Chroma(
                collection_name="my_collection",
                embedding_function=embeddings,
                persist_directory=chroma_db_path,
            )
            # Add documents to the ChromaDB collection
            vector_db.add_documents(documents)
            vector_db.persist()
            print(f"Data stored in ChromaDB at {chroma_db_path}.")
        except Exception as e:
            raise RuntimeError(f"Error handling ChromaDB database: {e}")

    else:
        raise ValueError("Unsupported db_type. Choose either 'faiss' or 'chroma'.")

# Example usage
data = [
    {"metadata": {"id": 1, "category": "news"}, "chunk_text": "This is the first text chunk."},
    {"metadata": {"id": 2, "category": "blog"}, "chunk_text": "Here is another chunk of text."},
    {"metadata": {"id": 3, "category": "report"}, "chunk_text": "The final text chunk in the list."}
]

try:
    # Store in FAISS
    store_in_vector_db_langchain(data, db_type="faiss", faiss_index_path="faiss_index")

    # Store in ChromaDB
    store_in_vector_db_langchain(data, db_type="chroma", chroma_db_path="chroma_storage")
except Exception as e:
    print(f"Error: {e}")




from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings import OpenAIEmbeddings
import os

def retrieve_top_k_results_sorted(
    query,
    k=5,
    faiss_index_path="faiss_index",
    chroma_db_path="chroma_storage"
):
    """
    Retrieves top K results from FAISS. If not available, falls back to ChromaDB.
    Results are sorted in descending order of similarity.

    Args:
        query (str or list): Query string or list of query strings.
        k (int): Number of top results to retrieve.
        faiss_index_path (str): Path to the FAISS index.
        chroma_db_path (str): Path to ChromaDB storage.

    Returns:
        list: List of dictionaries containing 'chunk_text', 'metadata', and 'similarity' scores, sorted by similarity.

    Raises:
        ValueError: If input query is invalid.
        RuntimeError: If both FAISS and ChromaDB fail to return results.
    """
    # Validate query input
    if not isinstance(query, (str, list)):
        raise ValueError("Query must be a string or a list of strings.")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")

    # Load embeddings
    embeddings = OpenAIEmbeddings()  # Replace with your desired embedding model

    # Helper function to perform similarity search
    def similarity_search(vector_db, query_list, k):
        results = []
        for q in query_list:
            docs_and_scores = vector_db.similarity_search_with_score(q, k=k)
            for doc, score in docs_and_scores:
                results.append({
                    "chunk_text": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity": score,
                })
        return results

    # Normalize query input to a list
    query_list = [query] if isinstance(query, str) else query

    # Attempt retrieval from FAISS
    try:
        if os.path.exists(faiss_index_path):
            vector_db_faiss = FAISS.load_local(faiss_index_path, embeddings)
            faiss_results = similarity_search(vector_db_faiss, query_list, k)
            if faiss_results:
                print("Results retrieved from FAISS.")
                # Sort by similarity in descending order
                return sorted(faiss_results, key=lambda x: x["similarity"], reverse=True)
    except Exception as e:
        print(f"Error with FAISS retrieval: {e}")

    # Fallback to ChromaDB
    try:
        vector_db_chroma = Chroma(
            collection_name="my_collection",
            embedding_function=embeddings,
            persist_directory=chroma_db_path,
        )
        chroma_results = similarity_search(vector_db_chroma, query_list, k)
        if chroma_results:
            print("Results retrieved from ChromaDB.")
            # Sort by similarity in descending order
            return sorted(chroma_results, key=lambda x: x["similarity"], reverse=True)
        else:
            print("No results found in ChromaDB either.")
    except Exception as e:
        print(f"Error with ChromaDB retrieval: {e}")

    # If no results from either database
    raise RuntimeError("No results found in both FAISS and ChromaDB.")

# Example usage
if __name__ == "__main__":
    query = "What is the capital of France?"
    try:
        results = retrieve_top_k_results_sorted(query, k=3, faiss_index_path="faiss_index", chroma_db_path="chroma_storage")
        print("Retrieved Results (sorted by similarity):", results)
    except Exception as e:
        print("Error:", e)


from transformers import pipeline

def generate_answer(template):
    """
    Generates an answer using the Mistral model based on the input template.

    Args:
        template (dict): A dictionary with the following fields:
            - query (str): The user query.
            - context (str): Contextual information for the query.
            - chat_history (list): List of previous interactions as strings.
            - instructions (str): Additional instructions for the model.

    Returns:
        str: The generated response from the LLM.
    
    Raises:
        ValueError: If any required field in the template is missing or invalid.
    """
    # Validate input template
    required_fields = ["query", "context", "chat_history", "instructions"]
    for field in required_fields:
        if field not in template:
            raise ValueError(f"Missing required field: {field}")
        if not isinstance(template[field], (str, list)):
            raise ValueError(f"Invalid type for field '{field}'. Expected str or list.")

    # Combine fields into a structured prompt
    prompt = (
        f"### Instructions:\n{template['instructions']}\n\n"
        f"### Context:\n{template['context']}\n\n"
        f"### Chat History:\n"
        f"{' '.join(template['chat_history']) if isinstance(template['chat_history'], list) else template['chat_history']}\n\n"
        f"### Query:\n{template['query']}\n\n"
        f"### Answer:"
    )

    # Load the LLM pipeline (Mistral)
    try:
        model_pipeline = pipeline("text-generation", model="mistral-7b")  # Replace with your Mistral model
        response = model_pipeline(prompt, max_length=512, num_return_sequences=1)
        answer = response[0]["generated_text"].split("### Answer:")[-1].strip()
        return answer
    except Exception as e:
        raise RuntimeError(f"Error generating response: {e}")

# Example usage
template = {
    "query": "What is the capital of France?",
    "context": "This is a geography-based question.",
    "chat_history": ["User: What is the capital of Germany?", "Assistant: The capital of Germany is Berlin."],
    "instructions": "Answer concisely and accurately. Provide the name of the capital city only."
}

try:
    answer = generate_answer(template)
    print("Generated Answer:", answer)
except Exception as e:
    print("Error:", e)		
	
	
