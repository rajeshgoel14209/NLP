from langchain.prompts import PromptTemplate

# Define the prompt template
template = """
You are an intelligent assistant. Your job is to extract specific IDs from the user's question. 
The IDs are numeric or alphanumeric strings, and they may be embedded within the question.
If no IDs are found, respond with "No IDs found."

Instruction: Extract all IDs from the following question.

User Question: "{question}"

Your Answer:
"""

# Create the PromptTemplate
id_extraction_prompt = PromptTemplate(
    input_variables=["question"],  # Variables to pass to the prompt
    template=template
)

# Example usage
user_question = "Can you retrieve the details for IDs 12345, AB6789, and XYZ123?"
formatted_prompt = id_extraction_prompt.format(question=user_question)

print("Generated Prompt:")
print(formatted_prompt)



from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load the model (e.g., Mistral)
def load_llm():
    model_name = "mistral-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    text_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=256)
    return HuggingFacePipeline(pipeline=text_pipeline)

# Create the LLM instance
llm = load_llm()

# Create the LLMChain
chain = LLMChain(llm=llm, prompt=id_extraction_prompt)

# Run the chain with a user question
response = chain.run(question=user_question)

print("LLM Response:")
print(response)
