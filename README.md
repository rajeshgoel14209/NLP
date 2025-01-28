# Function to chunk text
def chunk_text(text, max_tokens=512):
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield " ".join(words[i:i + max_tokens])

# Query and long document
query = "What is the capital of France?"
long_document = "..."  # Very long document

# Chunk the document
chunks = list(chunk_text(long_document, max_tokens=500))

# Pair query with each chunk
inputs = [(query, chunk) for chunk in chunks]

# Get similarity scores for each chunk
scores = model.predict(inputs)

# Take the chunk with the highest score
best_score = max(scores)
best_chunk_index = scores.index(best_score)
best_chunk = chunks[best_chunk_index]
