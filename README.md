1. Query Expansion
Synonym Expansion: Use WordNet or LLMs to add synonyms to the query.

Keyword Augmentation: Generate similar phrases using NLP models.

Embedding Expansion: Use multiple embeddings for query variations.

2. Hybrid Search (Dense + Sparse Retrieval)
Combine semantic search (vector embeddings) with BM25 (keyword-based retrieval) to improve accuracy.

Use weighted fusion of results from ChromaDB and BM25.

3. Contextual Embeddings
Use a cross-encoder to re-rank results by relevance.

Use query rewriting models (T5, GPT) to generate better queries.

4. Adaptive Search Parameters
Adjust similarity threshold to avoid overly generic results.

Increase top-k retrieval and re-rank results.

5. Metadata Filtering
Use metadata tags (e.g., categories, timestamps) to refine retrieval.

Implement hybrid filtering (e.g., keyword + vector search).

6. Query Preprocessing
Convert queries to lowercase and remove stopwords only if needed.

Use lemmatization instead of stemming for better matching.
