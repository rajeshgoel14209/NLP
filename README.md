https://ai.plainenglish.io/table-extraction-using-llms-unlocking-structured-data-from-documents-50cf21c98509

https://towardsdatascience.com/5-proven-query-translation-techniques-to-boost-your-rag-performance-47db12efe971

https://blog.stackademic.com/late-chunking-embedding-first-chunk-later-long-context-retrieval-in-rag-applications-3a292f6443bb

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Load Mistral model
model_name = "mistralai/Mistral-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Request format
class RequestData(BaseModel):
    model: str
    messages: list

@app.post("/v1/chat/completions")
async def chat_completion(request: RequestData):
    prompt = request.messages[-1]["content"]  # Extract latest message
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=512)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": request.model,
        "choices": [{"message": {"role": "assistant", "content": response_text}}],
        "usage": {"prompt_tokens": len(inputs["input_ids"][0]), "completion_tokens": len(outputs[0])}
    }


    from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(
    openai_api_base="http://your-server-ip:8000/v1",
    openai_api_key="your-api-key",  # Can be a placeholder if not required
)

response = llm.invoke("What is RAG?")
print(response)
