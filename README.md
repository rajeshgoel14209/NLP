import pdfplumber

def extract_landscape_table(pdf_path):
    """
    Extracts tables from landscape-oriented pages in a PDF document.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        list: A list of tables, where each table is represented as a list of rows (lists).
    """
    tables = []

    # Open the PDF
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            # Check page rotation
            rotation = page.rotation or 0
            print(f"Page {page_number}: Rotation = {rotation}°")

            # Correct the orientation if rotated
            if rotation in [90, 270]:
                print(f"Normalizing orientation for Page {page_number}")
                page = page.rotate(-rotation)  # Rotate back to 0 degrees

            # Extract tables
            page_tables = page.extract_tables()
            for table in page_tables:
                tables.append(table)

    return tables

# Example usage
pdf_path = "example.pdf"  # Replace with the path to your PDF file
tables = extract_landscape_table(pdf_path)

# Print extracted tables
for i, table in enumerate(tables, start=1):
    print(f"\nTable {i}:")
    for row in table:
        print(row)


import pandas as pd

for i, table in enumerate(tables, start=1):
    df = pd.DataFrame(table[1:], columns=table[0])  # Assuming the first row is the header
    df.to_csv(f"table_{i}.csv", index=False)
        
