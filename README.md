https://ai.plainenglish.io/table-extraction-using-llms-unlocking-structured-data-from-documents-50cf21c98509

https://towardsdatascience.com/5-proven-query-translation-techniques-to-boost-your-rag-performance-47db12efe971

https://blog.stackademic.com/late-chunking-embedding-first-chunk-later-long-context-retrieval-in-rag-applications-3a292f6443bb

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load pre-trained model and tokenizer (you can use any transformer-based model for Cross-Encoder)
model_name = "sentence-transformers/msmarco-distilbert-base-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Query embedding and document embeddings
query_embedding = "your_query_embedding_here"  # Replace with your actual query embedding
document_embeddings = ["doc1", "doc2", "doc3"]  # List of document texts

# Tokenizing the query and document pairs
def prepare_input(query, doc):
    # Create pairs of query and document as input for the Cross-Encoder
    inputs = tokenizer(query, doc, return_tensors='pt', padding=True, truncation=True)
    return inputs

# Evaluate the relevance of each document to the query
def get_relevance_scores(query, documents):
    scores = []
    for doc in documents:
        inputs = prepare_input(query, doc)
        with torch.no_grad():
            output = model(**inputs)
        score = output.logits.squeeze().item()  # Assuming logits represent relevance score
        scores.append(score)
    return scores

# Example usage
query = "What is the capital of France?"
relevance_scores = get_relevance_scores(query, document_embeddings)
print(relevance_scores)
]

# Apply weighted mean pooling with equal weights
pooled_embedding = weighted_mean_pooling(embeddings_list, weighting_strategy="equal")
print("Pooled Embedding:", pooled_embedding)

