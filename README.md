import pandas as pd

# Example DataFrame
data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [30, 25, 35],
    "Occupation": ["Engineer", "Designer", "Teacher"]
}
df = pd.DataFrame(data)

# Convert DataFrame to a plain text table
def dataframe_to_plain_text(df):
    headers = df.columns.tolist()
    rows = df.values.tolist()
    col_widths = [max(len(str(item)) for item in [header] + list(df[col])) for header, col in zip(headers, df.columns)]
    row_format = " | ".join([f"{{:<{width}}}" for width in col_widths])
    header_line = row_format.format(*headers)
    separator_line = "-+-".join(["-" * width for width in col_widths])
    rows_text = "\n".join([row_format.format(*[str(cell) for cell in row]) for row in rows])
    return f"{header_line}\n{separator_line}\n{rows_text}"

table_text = dataframe_to_plain_text(df)


import openai

response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt,
    max_tokens=100
)






from sentence_transformers import SentenceTransformer
import faiss

# Step 1: Embed the DataFrame rows as text entries
model = SentenceTransformer("all-MiniLM-L6-v2")
data_rows = df.apply(lambda row: " | ".join(row.astype(str)), axis=1).tolist()
embeddings = model.encode(data_rows)

# Step 2: Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Step 3: Retrieve relevant rows for a query
query_embedding = model.encode(["oldest person in the table"])[0]
_, indices = index.search(query_embedding.reshape(1, -1), k=1)

# Step 4: Use retrieved row(s) in the LLM prompt
retrieved_row = data_rows[indices[0][0]]
retrieved_prompt = f"""
Below is a row of data related to your query:

{retrieved_row}

Please answer the query: {query}
"""

