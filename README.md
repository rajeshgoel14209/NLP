from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# Load the BERT-based model and tokenizer
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"  # Pretrained BERT for QA
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)


# Create a QA pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

def answer_question(question, context):
    result = qa_pipeline({
        "question": question,
        "context": context
    })
    return result["answer"]

# Example Context
context = """
The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. 
It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
"""

# Example Question
question = "Who designed the Eiffel Tower?"

# Get the Answer
answer = answer_question(question, context)
print(f"Question: {question}")
print(f"Answer: {answer}")


