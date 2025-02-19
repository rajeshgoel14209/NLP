from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load LLaMA 2-7B OpenLLaMA model
model_name = "openlm-research/open_llama_7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

def llama_generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_length=512)
    return tokenizer.decode(output[0], skip_special_tokens=True)


from llama_index.tools import QueryEngineTool
from llama_index.query_engine import PandasQueryEngine
import pandas as pd

df = pd.read_csv("data.csv")
query_engine_pandas = PandasQueryEngine(df)

pandas_tool = QueryEngineTool(
    query_engine=query_engine_pandas,
    metadata={"name": "pandas_tool", "description": "Queries structured data from a CSV file"}
)


from llama_index.tools import WikipediaQueryEngine

wiki_engine = WikipediaQueryEngine()

wiki_tool = QueryEngineTool(
    query_engine=wiki_engine,
    metadata={"name": "wiki_tool", "description": "Fetches information from Wikipedia"}
)


from llama_index.vector_stores import ChromaVectorStore
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.storage.storage_context import StorageContext
from llama_index.retrievers import VectorIndexRetriever
from llama_index import VectorStoreIndex, SimpleDirectoryReader

# Load documents into ChromaDB
documents = SimpleDirectoryReader("docs/").load_data()
chroma_store = ChromaVectorStore(persist_dir="./chroma_db")
storage_context = StorageContext.from_defaults(vector_store=chroma_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
query_engine_chroma = RetrieverQueryEngine(retriever=retriever)

chroma_tool = QueryEngineTool(
    query_engine=query_engine_chroma,
    metadata={"name": "chroma_tool", "description": "Retrieves context from ChromaDB for RAG"}
)

4. Register Tools and Create an Agent

from llama_index.agent import OpenAIAgent

class MistralLLM:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

    def complete(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        output = self.model.generate(**inputs, max_length=512)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

# Initialize Mistral LLM
mistral_llm = MistralLLM()

# Create agent with tools
agent = OpenAIAgent.from_tools([pandas_tool, wiki_tool, chroma_tool], llm=mistral_llm)

# Query the agent
response = agent.query("What is the GDP of India in 2023?")
print(response)
