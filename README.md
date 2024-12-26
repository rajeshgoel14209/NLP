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
