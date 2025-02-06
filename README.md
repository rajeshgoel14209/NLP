from langchain.agents import ZeroShotAgent
from langchain.prompts import PromptTemplate

custom_prompt_with_history = PromptTemplate(
    template="""
You are an AI agent that assists users by reasoning step by step and using tools when necessary.

## Chat History:
{chat_history}

## Instructions:
- Think before you act.
- Use tools only when necessary.
- Respond using the structured format:

### Example:
Thought: "I need to search for AI research."
Action: search[{"query": "latest AI research"}]
Observation: "AI research articles found."
Final Answer: "Here are the latest AI research articles: ..."

## Available Tools:
{tools}

## Current Question:
{input}
{agent_scratchpad}
""",
    input_variables=["tools", "input", "agent_scratchpad", "chat_history"],
)


from collections import deque

class ChatMemory:
    def __init__(self, max_length=5):
        self.memory = deque(maxlen=max_length)

    def add_interaction(self, user_input, agent_response):
        self.memory.append(f"User: {user_input}\nAI: {agent_response}")

    def get_history(self):
        return "\n".join(self.memory)


  from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# Initialize chat memory
chat_memory = ChatMemory(max_length=5)

# Sample tool
def search_tool(query: str):
    return f"Searching for: {query}..."

search = Tool(name="search", func=search_tool, description="Searches for online information.")

# Load Mistral LLM
llm = ChatOpenAI(model="mistral", openai_api_key="YOUR_API_KEY")

# Run a conversation loop
while True:
    user_input = input("User: ")
    
    # Inject chat history into the agent prompt
    chat_history = chat_memory.get_history()
    
    # Initialize agent with custom prompt including chat history
    agent = initialize_agent(
        tools=[search],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs={"prompt": custom_prompt_with_history},
        verbose=True,
    )

    # Run the agent
    response = agent.run(user_input)

    # Store interaction in chat memory
    chat_memory.add_interaction(user_input, response)

    print(f"AI: {response}")      
