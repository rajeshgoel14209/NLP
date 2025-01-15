import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Step 1: Define Case ID Extraction Regex
CASE_ID_PATTERN = r"CASE-\d+"  # Adjust pattern as per your requirements

# Step 2: Load Mistral Model
def load_mistral_model():
    model_name = "mistral-7b"  # Replace with your desired Mistral model path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # HuggingFace pipeline for text generation
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # CPU only
        max_length=100,  # Short response
        temperature=0.7,
        top_p=0.9,
    )
    return text_pipeline

# Step 3: Extract Case IDs Using Regex
def extract_case_ids(query):
    # Find all case IDs using regex
    case_ids = re.findall(CASE_ID_PATTERN, query)
    return case_ids if case_ids else []

# Step 4: Process Query Using Mistral
def process_query_with_mistral(mistral_pipeline, query):
    case_ids = extract_case_ids(query)
    if not case_ids:
        return {"case_ids": [], "message": "No case IDs found."}

    # Create a refined query for Mistral
    mistral_prompt = (
        f"You are a smart assistant. Extract only case IDs from the input query: '{query}'. "
        f"Do not include unnecessary information. Return the IDs as a list."
    )

    # Generate response
    mistral_response = mistral_pipeline(mistral_prompt, max_length=50, num_return_sequences=1)
    refined_response = mistral_response[0]["generated_text"].strip()

    # Parse Mistral's output to refine the extracted IDs (optional)
    return {"case_ids": case_ids, "refined_response": refined_response}

# Step 5: Main Function to Run the Pipeline
def run_pipeline(query):
    mistral_pipeline = load_mistral_model()  # Load Mistral once for efficiency
    result = process_query_with_mistral(mistral_pipeline, query)
    return result

# Step 6: Example Usage
if __name__ == "__main__":
    # Example input query
    user_query = "Please check the status of CASE-12345 and CASE-67890. Also, ignore ID-11111."

    # Run the pipeline
    result = run_pipeline(user_query)
    print("Result:", result)
