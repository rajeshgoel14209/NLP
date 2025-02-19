pip install llama-index openai

from llama_index.tools import QueryEngineTool
from llama_index.query_engine import PandasQueryEngine
from llama_index.llms import OpenAI
import pandas as pd


df = pd.read_csv("your_data.csv")
query_engine = PandasQueryEngine(df)

query_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata={"name": "data_tool", "description": "Query the dataset for insights"}
)

from llama_index.agent import OpenAIAgent

llm = OpenAI(model="gpt-4")  # Use your preferred model
agent = OpenAIAgent.from_tools([query_tool], llm=llm)

response = agent.query("What is the average value of column X?")
print(response)

from llama_index.llms import CustomLLM
import requests

class MistralLLM(CustomLLM):
    def __init__(self, api_url: str):
        self.api_url = api_url

    def complete(self, prompt: str, **kwargs):
        response = requests.post(self.api_url, json={"prompt": prompt, "temperature": 0.7, "max_tokens": 512})
        return response.json()["response"]  # Modify based on your API response format

from llama_index.vector_stores import ChromaVectorStore
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.storage.storage_context import StorageContext
from llama_index.retrievers import VectorIndexRetriever
from llama_index import VectorStoreIndex, SimpleDirectoryReader

# Load documents
documents = SimpleDirectoryReader("data/").load_data()

# Create ChromaDB vector store
chroma_store = ChromaVectorStore(persist_dir="./chroma_db")
storage_context = StorageContext.from_defaults(vector_store=chroma_store)

# Index the documents
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Create a retriever and query engine
retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
query_engine = RetrieverQueryEngine(retriever=retriever)

from llama_index.tools import QueryEngineTool

query_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata={"name": "chroma_query_tool", "description": "Retrieves information from ChromaDB"}
)

from llama_index.agent import OpenAIAgent

# Initialize Mistral LLM
mistral_llm = MistralLLM(api_url="http://your-server:8000/completion")  # Change to your actual Mistral API URL

# Create the agent
agent = OpenAIAgent.from_tools([query_tool], llm=mistral_llm)

# Test the agent
response = agent.query("What does the document say about AI advancements?")
print(response)
