import re
import string
from collections import Counter
from typing import List, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score

def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation, and normalize whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)  # Remove articles
    s = re.sub(f"[{string.punctuation}]", "", s)  # Remove punctuation
    s = " ".join(s.split())  # Normalize whitespace
    return s

def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_f1_score(prediction: str, ground_truth: str) -> float:
    """Compute F1-score (token overlap)."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

def evaluate_rag_pipeline(questions: List[str], actual_answers: List[str], rag_pipeline) -> dict:
    """Evaluates RAG pipeline accuracy using EM and F1-score."""
    em_scores, f1_scores = [], []

    for question, ground_truth in zip(questions, actual_answers):
        retrieved_answer = rag_pipeline(question)  # Call your RAG model

        em = compute_exact_match(retrieved_answer, ground_truth)
        f1 = compute_f1_score(retrieved_answer, ground_truth)

        em_scores.append(em)
        f1_scores.append(f1)

        print(f"Q: {question}\nGround Truth: {ground_truth}\nPredicted: {retrieved_answer}\nEM: {em:.2f}, F1: {f1:.2f}\n")

    return {
        "Exact Match Accuracy": sum(em_scores) / len(em_scores),
        "F1 Score": sum(f1_scores) / len(f1_scores)
    }

# Example Usage
if __name__ == "__main__":
    def mock_rag_pipeline(question: str) -> str:
        """Replace this with your actual RAG pipeline function."""
        return "Mock answer for: " + question

    test_questions = ["What is the capital of France?", "Who wrote 1984?"]
    actual_answers = ["Paris", "George Orwell"]

    results = evaluate_rag_pipeline(test_questions, actual_answers, mock_rag_pipeline)
    print("Final Evaluation Results:", results)
