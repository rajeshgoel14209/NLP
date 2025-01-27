import fitz  # PyMuPDF

def extract_word_styling(pdf_path):
    try:
        word_styling = []

        # Open the PDF
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]  # Extract text blocks as dictionary

            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        # Split spans into words to analyze styling word by word
                        words = span["text"].split()
                        for word in words:
                            word_data = {
                                "page": page_num + 1,
                                "text": word,
                                "font": span.get("font"),  # Font name
                                "font_size": span.get("size"),  # Font size
                                "font_color": fitz.get_color_string(span.get("color")),  # Font color in RGB
                                "background_color": None,  # Background colors are not directly supported
                                "alignment": block.get("type", "unknown"),  # Block alignment
                                "bounding_box": span.get("bbox"),  # Bounding box for word
                            }
                            word_styling.append(word_data)

        return word_styling

    except Exception as e:
        print(f"Error while extracting word styling: {e}")
        return []

# Example Usage
pdf_file_path = "example.pdf"
styling_data = extract_word_styling(pdf_file_path)

# Display results
for item in styling_data[:10]:  # Show first 10 results
    print(item)
