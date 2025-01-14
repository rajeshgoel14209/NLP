from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import numpy as np
import os

# Step 1: Load Mistral Model for Question Answering
def load_mistral_model():
    model_name = "mistral-7b"  # Replace with the specific model name/path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Use HuggingFace pipeline
    qa_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.7,
        top_p=0.9,
    )

    # Wrap it in LangChain's HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=qa_pipeline)
    return llm

# Step 2: Load or Create a Vector Database
def create_or_load_vector_db(embedding_model_name, db_path="faiss_index"):
    # Use HuggingFace embeddings for vectorization
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # Check if FAISS index exists
    if os.path.exists(db_path):
        print(f"Loading existing FAISS index from {db_path}...")
        vector_db = FAISS.load_local(db_path, embeddings)
    else:
        print("Creating a new FAISS index...")
        # Example documents
        docs = [
            {"chunk_text": "The Eiffel Tower is in Paris.", "metadata": {"source": "tourism"}},
            {"chunk_text": "Python is a popular programming language.", "metadata": {"source": "tech"}},
            {"chunk_text": "Mistral is a powerful language model.", "metadata": {"source": "ai"}},
        ]

        # Format documents for LangChain
        documents = [
            {"page_content": doc["chunk_text"], "metadata": doc["metadata"]} for doc in docs
        ]

        # Create FAISS index
        vector_db = FAISS.from_documents(documents, embeddings)

        # Save the index for future use
        vector_db.save_local(db_path)

    return vector_db

# Step 3: Define RAG Pipeline
def create_rag_pipeline(llm, vector_db):
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = RetrievalQA(llm=llm, retriever=retriever)
    return qa_chain

# Step 4: Ask Questions Using the RAG Pipeline
def ask_question(qa_chain, query):
    try:
        response = qa_chain.run(query)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

# Main Execution
if __name__ == "__main__":
    # Initialize Mistral LLM
    llm = load_mistral_model()

    # Create or load a vector database
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Replace with your embedding model
    vector_db = create_or_load_vector_db(embedding_model_name)

    # Create RAG pipeline
    rag_pipeline = create_rag_pipeline(llm, vector_db)

    # Query the RAG pipeline
    question = "Where is the Eiffel Tower located?"
    answer = ask_question(rag_pipeline, question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
