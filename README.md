pip install tabula-py pdf2image pytesseract opencv-python
pattern = r"\b(cover\s*page|executive\s*summary|sources?\s*of\s*repayment|financial\s*analysis|exposure\s*analysis|appendix)\b"
import cv2
import pytesseract
from pdf2image import convert_from_path
import tabula

# Convert image to PDF
def image_to_pdf(image_path, output_pdf):
    images = convert_from_path(image_path)
    images[0].save(output_pdf, "PDF")

# Extract tables using Tabula
def extract_tables_from_pdf(pdf_path):
    tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
    return tables

# Main process
image_path = "table_image.jpg"  # Replace with your image path
pdf_path = "output.pdf"

image_to_pdf(image_path, pdf_path)
tables = extract_tables_from_pdf(pdf_path)

# Print extracted tables
for idx, table in enumerate(tables):
    print(f"Table {idx+1}:\n", table)

pip install camelot-py[cv]

import camelot

# Convert Image to PDF first (same as previous example)

# Extract tables using Camelot
def extract_tables_with_camelot(pdf_path):
    tables = camelot.read_pdf(pdf_path, pages="1")
    return tables

# Extract tables
tables = extract_tables_with_camelot("output.pdf")

# Print extracted tables
for i, table in enumerate(tables):
    print(f"Table {i+1}:\n", table.df)


import cv2
import pytesseract
import pandas as pd

# Read the image
image = cv2.imread("table_image.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply OCR
custom_config = r"--oem 3 --psm 6"
text = pytesseract.image_to_string(gray, config=custom_config)

# Print raw extracted text
print("Extracted Text:\n", text)

    
