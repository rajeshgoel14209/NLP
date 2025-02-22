import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# Load GTE model for embedding generation
EMBEDDING_MODEL = "thenlper/gte-large"  # You can replace it with a smaller variant if needed
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
embed_model = AutoModel.from_pretrained(EMBEDDING_MODEL)

# Function to generate embeddings
def generate_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # Extract CLS token representation

# Example Text & Query
text = "Machine learning models require a lot of training data."
query = "Do ML models need large datasets?"

# Generate embeddings for text and query
text_embedding = generate_embedding(text)
query_embedding = generate_embedding(query)

# Concatenate both embeddings
combined_embedding = torch.cat((text_embedding, query_embedding), dim=1)

# Define a simple Cross-Encoder model
class CrossEncoder(nn.Module):
    def __init__(self, input_dim):
        super(CrossEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),  # Reduce dimensions
            nn.ReLU(),
            nn.Linear(128, 1)  # Output relevance score
        )
    
    def forward(self, x):
        return self.fc(x)

# Initialize the cross-encoder
input_dim = combined_embedding.shape[1]  # Double the embedding size since we concatenate
cross_encoder = CrossEncoder(input_dim)

# Evaluate relevance score
with torch.no_grad():
    score = cross_encoder(combined_embedding)

print(f"Relevance Score: {score.item()}")



#####################################################################################################


import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# Load GTE model for embedding generation
EMBEDDING_MODEL = "thenlper/gte-large"  # Change to "gte-base" or "gte-small" if needed
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
embed_model = AutoModel.from_pretrained(EMBEDDING_MODEL)

# Function to generate embeddings using GTE
def generate_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # Extract CLS token representation

# Define a simple Cross-Encoder for Reranking
class CrossEncoder(nn.Module):
    def __init__(self, input_dim):
        super(CrossEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),  # Reduce dimensions
            nn.ReLU(),
            nn.Linear(128, 1)  # Output relevance score
        )
    
    def forward(self, x):
        return self.fc(x)

# Example query and candidate passages
query = "What are the benefits of machine learning?"
candidates = [
    "Machine learning helps in data-driven decision making.",
    "Deep learning is a subset of machine learning.",
    "Quantum computing is an emerging technology.",
    "Supervised learning requires labeled datasets."
]

# Generate embeddings for query and candidates
query_embedding = generate_embedding(query)
candidate_embeddings = [generate_embedding(text) for text in candidates]

# Concatenate query and candidate embeddings for cross-encoding
combined_embeddings = [torch.cat((query_embedding, candidate_embedding), dim=1) for candidate_embedding in candidate_embeddings]
combined_embeddings = torch.vstack(combined_embeddings)

# Initialize cross-encoder model
input_dim = combined_embeddings.shape[1]  # Double the embedding size
cross_encoder = CrossEncoder(input_dim)

# Compute relevance scores
with torch.no_grad():
    scores = cross_encoder(combined_embeddings).squeeze()

# Rank candidates based on scores
ranked_results = sorted(zip(candidates, scores.tolist()), key=lambda x: x[1], reverse=True)

# Print ranked results
print("\nRanked Candidates:")
for rank, (text, score) in enumerate(ranked_results, 1):
    print(f"{rank}. [{score:.4f}] {text}")


  ##################################################################################################################


  import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np

# Load GTE model for embedding generation
EMBEDDING_MODEL = "thenlper/gte-large"  # Replace with a smaller variant if needed
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
embed_model = AutoModel.from_pretrained(EMBEDDING_MODEL)

# Function to generate embeddings
def generate_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # Extract CLS token representation

# Define a simple Cross-Encoder for Reranking
class CrossEncoder(nn.Module):
    def __init__(self, input_dim):
        super(CrossEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),  # Reduce dimensions
            nn.ReLU(),
            nn.Linear(128, 1)  # Output relevance score
        )
    
    def forward(self, x):
        return self.fc(x)

# Example Query
query = "What are the benefits of machine learning?"

# Example Candidate Passages
candidates = [
    "Machine learning allows computers to improve automatically through experience.",
    "Deep learning is a subset of AI that focuses on neural networks.",
    "Machine learning is widely used in healthcare for predicting diseases.",
    "Physics is the study of matter and energy.",
]

# Generate embeddings for the query
query_embedding = generate_embedding(query)

# Generate embeddings for each candidate passage
candidate_embeddings = [generate_embedding(text) for text in candidates]

# Concatenate query and candidate embeddings
paired_embeddings = [torch.cat((query_embedding, text_embedding), dim=1) for text_embedding in candidate_embeddings]

# Initialize the cross-encoder
input_dim = paired_embeddings[0].shape[1]  # Double the embedding size
cross_encoder = CrossEncoder(input_dim)

# Compute relevance scores
with torch.no_grad():
    scores = [cross_encoder(embed).item() for embed in paired_embeddings]

# Sort candidates by descending score (higher means more relevant)
sorted_results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

# Display results
print("\n*** Reranked Results ***\n")
for rank, (text, score) in enumerate(sorted_results, 1):
    print(f"{rank}. {text} (Score: {score:.4f})")


#########################################################################################

import jiwer
import nltk
import torch
import gensim.downloader as api
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from bert_score import score
from gensim.models import KeyedVectors
from scipy.spatial.distance import cdist

# Download necessary NLTK resources
nltk.download("wordnet")
nltk.download("omw-1.4")

# Load a pre-trained Word2Vec model for Word Movers’ Distance (WMD)
word_vectors = api.load("word2vec-google-news-300")  # Uses Google's Word2Vec model

# Example Responses
actual_response = "Machine learning helps in predictive analytics by analyzing data patterns."
llm_response = "Predictive analytics is improved by machine learning through data pattern analysis."

# 1. **Word Error Rate (WER)**
wer = jiwer.wer(actual_response, llm_response)

# 2. **BLEU Score**
actual_tokens = [actual_response.split()]
llm_tokens = llm_response.split()
bleu_score = sentence_bleu(actual_tokens, llm_tokens)

# 3. **ROUGE Scores**
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
rouge_scores = scorer.score(actual_response, llm_response)

# 4. **METEOR Score**
meteor = meteor_score([actual_response.split()], llm_response.split())

# 5. **BERTScore**
P, R, F1 = score([llm_response], [actual_response], lang="en")
bert_f1 = F1.mean().item()

# 6. **Word Movers' Distance (WMD)**
def wmd_distance(sent1, sent2):
    sent1 = [word for word in sent1.lower().split() if word in word_vectors]
    sent2 = [word for word in sent2.lower().split() if word in word_vectors]
    
    if not sent1 or not sent2:
        return float("inf")  # Return a high distance if no valid words

    return word_vectors.wmdistance(sent1, sent2)

wmd_score = wmd_distance(actual_response, llm_response)

# Print all scores
print(f"Word Error Rate (WER): {wer:.4f}")
print(f"BLEU Score: {bleu_score:.4f}")
print(f"ROUGE-1: {rouge_scores['rouge1'].fmeasure:.4f}")
print(f"ROUGE-2: {rouge_scores['rouge2'].fmeasure:.4f}")
print(f"ROUGE-L: {rouge_scores['rougeL'].fmeasure:.4f}")
print(f"METEOR Score: {meteor:.4f}")
print(f"BERTScore (F1): {bert_f1:.4f}")
print(f"Word Movers' Distance (WMD): {wmd_score:.4f}")
