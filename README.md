https://www.kaggle.com/datasets/gazal5277/e-commerce-product-and-customer-dataset?phase=FinishSSORegistration&returnUrl=/datasets/gazal5277/e-commerce-product-and-customer-dataset/versions/1?resource=download&SSORegistrationToken=CfDJ8PHSCL9k9s1HuJ2cRFBFhuj7jj2GrKbxP6UAmI61_-zw4tdXmuoUsxSLqImfSrYJgsHl6_EmqAEAV2q4ZOiuVq9-HPRZJzSCT4qJOTfwEQPwOsxoRZT8wqxWBGlNZT50mexINu0CX5TuQU2IyBSoNp6bW80UmWoTNhq77BPe3CWQTWaOw3Ykpqe8CHkUAJ-p3kx9HlaBmk_Io0t5Bc0rfmBo8_UtvJp180-7eYKdVFcCGRMiYSuXRyYLHaWetgiEKp81MYz6s-txUQWJ-qUC5sVYqKWrU3ZyEIC0NRR_YHUIfEXvN99siGdam9QFTKldkid1rqe9FU5X1HwZT1mKXZSXKqV6sg&DisplayName=Rajesh%20Goel

from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://your-gpu-server:8000/v1",
    model="mistral",
    temperature=0
)


from langchain.tools import Tool
from langchain.utilities import WikipediaAPIWrapper

wiki = WikipediaAPIWrapper()
wiki_tool = Tool(
    name="WikipediaSearch",
    func=wiki.run,
    description="Search Wikipedia for general knowledge queries."
)
tools = [wiki_tool]


from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain.agents.agent import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import re

# Custom Prompt
prompt_template = """You are a helpful AI agent. Given the input question, you must determine the best action using the available tools.

Question: {input}
Thought: {agent_scratchpad}
"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["input", "agent_scratchpad"]
)

# Custom Output Parser
class CustomOutputParser(AgentOutputParser):
    def parse(self, text: str):
        match = re.search(r"Action: (.+?)\nAction Input: (.+)", text, re.DOTALL)
        if match:
            return AgentAction(tool=match.group(1), tool_input=match.group(2), log=text)
        else:
            return AgentFinish(return_values={"output": text}, log=text)

output_parser = CustomOutputParser()

# Custom Agent
class CustomAgent(LLMSingleActionAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

agent = CustomAgent(
    llm_chain=llm,
    output_parser=output_parser,
    stop_sequence=["\n"],
)

# Agent Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)


response = agent_executor.run("Who is Albert Einstein?")
print(response)


