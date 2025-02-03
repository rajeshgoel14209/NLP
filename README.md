from langchain.agents import initialize_agent, Tool
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory

# Define retrieval function
def search_vectorstore(query):
    retriever = vectorstore.as_retriever()
    return retriever.get_relevant_documents(query)

# Define tools (search + LLM)
tools = [
    Tool(name="Vector Search", func=search_vectorstore, description="Search for relevant documents."),
    Tool(name="Mistral Model", func=mistral_generate, description="Generate text using Mistral model.")
]

# Initialize Agent
memory = ConversationBufferMemory(memory_key="chat_history")
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", memory=memory, verbose=True)



query = "How is AI transforming industries?"
response = agent.run(query)
print("Agent Response:", response)
