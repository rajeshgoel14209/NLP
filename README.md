system_prompt = """
You are an AI agent that follows a strict format. After providing a "Thought", you must always take an "Action".
Respond only in JSON format like this:

{
  "thought": "I need to search for the latest AI trends.",
  "action": "search",
  "parameters": {"query": "latest AI trends"}
}

Do not add any explanations or extra text. Just return valid JSON.
"""


import json
import re

def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return json.loads(match.group()) if match else None

llm_response = 'Thought: I should look up AI news.\n```json\n{"action": "search", "parameters": {"query": "AI news"}}\n```'
parsed_output = extract_json(llm_response)
print(parsed_output)  # Should return a dictionary


from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# Define a sample tool
def search_tool(query: str):
    return f"Searching for {query}..."

search = Tool(name="search", func=search_tool, description="Searches for information online.")

# Initialize Mistral with function calling
llm = ChatOpenAI(model="mistral", openai_api_key="YOUR_API_KEY")

agent = initialize_agent(
    tools=[search],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,  # Forces function calling behavior
    verbose=True,
)

agent_executor = initialize_agent(
    tools=[search],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True  # Enables debugging output
)

agent_executor.run("Find the latest AI trends.")



agent.run("Find the latest AI research.")
