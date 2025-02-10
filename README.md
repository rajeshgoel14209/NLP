1. Create a Custom LLM Wrapper for LangChain

2. from langchain.llms.base import LLM
from typing import Any, List, Optional

class CustomLLM(LLM):
    """Custom LLM wrapper for integrating with Mistral Agent."""
    
    def __init__(self, api_endpoint: str, api_key: Optional[str] = None):
        super().__init__()
        self.api_endpoint = api_endpoint
        self.api_key = api_key  # If needed for authentication

    @property
    def _llm_type(self) -> str:
        return "custom_llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Sends the prompt to the custom LLM API and returns the response."""
        import requests
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = requests.post(
            self.api_endpoint, json={"prompt": prompt, "stop": stop}, headers=headers
        )
        return response.json().get("text", "")


from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool

# Initialize custom LLM
custom_llm = CustomLLM(api_endpoint="http://localhost:8000/generate")

# Define a sample tool
def sample_tool(query: str) -> str:
    return f"Processed query: {query}"

tool = Tool(
    name="SampleTool",
    func=sample_tool,
    description="A tool that processes a given query."
)

# Create an agent using the custom LLM
agent_executor = initialize_agent(
    tools=[tool],
    llm=custom_llm,  # Use the custom LLM here
    agent=AgentType.OPENAI_FUNCTIONS,  # Mistral-style agent
    verbose=True
)

# Run the agent
response = agent_executor.run("Tell me about LangChain?")
print(response)


3. Verify and Test
Start your custom LLM server (e.g., via FastAPI).
Ensure your custom LLM endpoint is accessible.
Run the agent script and check if the Mistral agent properly calls your custom LLM.
