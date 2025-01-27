import pdfplumber

def extract_text_styling(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text_styling_details = []
            for page_num, page in enumerate(pdf.pages):
                for char in page.chars:
                    text_styling_details.append({
                        "page": page_num + 1,
                        "text": char.get("text"),
                        "font": char.get("fontname"),
                        "font_size": char.get("size"),
                        "x": char.get("x0"),
                        "y": char.get("top"),
                        "width": char.get("x1") - char.get("x0"),
                        "height": char.get("bottom") - char.get("top")
                    })
        return text_styling_details
    except Exception as e:
        print(f"Error while extracting text styling: {e}")
        return []

# Example Usage
pdf_file_path = "example.pdf"
styling_data = extract_text_styling(pdf_file_path)

# Print results
for item in styling_data[:10]:  # Limit output for brevity
    print(item)
