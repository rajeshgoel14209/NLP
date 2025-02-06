from langchain.agents import ZeroShotAgent
from langchain.prompts import PromptTemplate

# Custom Prompt
custom_prompt = PromptTemplate(
    template="""
You are an AI agent helping with tasks using available tools. Follow this process strictly:

1. Think about the best next step. Write this as:
   Thought: "..."
   
2. Decide which tool to use. Respond with:
   Action: tool_name[parameters]
   
3. Observe the tool's response.
   
4. If needed, repeat the process. Otherwise, provide:
   Final Answer: "..."

Here are the available tools:
{tools}

You must strictly follow this format.

Begin!

Question: {input}
{agent_scratchpad}
""",
    input_variables=["tools", "input", "agent_scratchpad"],
)


from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# Sample tool
def search_tool(query: str):
    return f"Searching for: {query}..."

search = Tool(name="search", func=search_tool, description="Searches for online information.")

# Load Mistral LLM
llm = ChatOpenAI(model="mistral", openai_api_key="YOUR_API_KEY")

# Create Zero-Shot ReAct agent with custom prompt
agent = initialize_agent(
    tools=[search],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    agent_kwargs={"prompt": custom_prompt},  # Custom prompt applied here
    verbose=True,
)

# Run agent
agent.run("Find the latest AI research.")

custom_prompt_json = PromptTemplate(
    template="""
You are an AI assistant. Always respond in valid JSON format:

{
  "thought": "Analyze the input...",
  "action": "tool_name",
  "parameters": {"query": "search term"}
}

Here are the available tools:
{tools}

Question: {input}
{agent_scratchpad}
""",
    input_variables=["tools", "input", "agent_scratchpad"],
)

custom_prompt_no_hallucination = PromptTemplate(
    template="""
You are an AI assistant with access to the following tools:

{tools}

Always respond in this format:
1. Thought: "..."
2. Action: Choose one of the above tools.
3. Parameters: Provide the correct parameters.

If you don’t know the answer, respond:
Final Answer: "I cannot complete this request."

Question: {input}
{agent_scratchpad}
""",
    input_variables=["tools", "input", "agent_scratchpad"],
)
