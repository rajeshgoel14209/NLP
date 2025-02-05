import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.agents import Tool, ZeroShotAgent, AgentExecutor

# Load the Mistral7B model from Hugging Face
model_id = "mistralai/Mistral-7B-Instruct-v0.1"  # Replace with your Mistral7B model if needed
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"  # This uses available GPUs; adjust if needed
)

# Create a text-generation pipeline for the model
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    do_sample=True,
    temperature=0.7,
)

# Wrap the pipeline with LangChain's HuggingFacePipeline LLM
llm = HuggingFacePipeline(pipeline=pipe)

# Define some example tools for the agent to use
tools = [
    Tool(
        name="Calculator",
        func=lambda x: str(eval(x)),  # simple calculator function; use with caution!
        description="Useful for performing arithmetic calculations."
    ),
    Tool(
        name="Search",
        func=lambda query: f"Simulated search results for query: '{query}'",
        description="Useful for searching for information."
    ),
]

# Define prompt prefixes and suffixes to guide the agent's behavior
prefix = (
    "You are an AI assistant that can answer questions and use tools if needed. "
    "When appropriate, decide to use one of the available tools to assist with your answer."
)
suffix = (
    "If you decide a tool is needed, make sure to call it appropriately. "
    "Otherwise, provide a direct answer to the question."
)

# Initialize the ZeroShotAgent with the Mistral LLM and the tools
agent = ZeroShotAgent(llm=llm, tools=tools, prefix=prefix, suffix=suffix)

# Create an AgentExecutor that ties the agent and tools together, with verbose logging
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

# Run a test query through the agent
query = "What is the result of 15 + 20, and can you search for the latest news on AI?"
result = agent_executor.run(query)

print("Agent Response:")
print(result)
