import pdfplumber
from PIL import ImageColor

def extract_word_styling(pdf_path):
    try:
        word_styling_details = []

        # Open the PDF
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                words = page.extract_words()
                for word in words:
                    # Extract characters in the word for styling details
                    chars = [
                        char for char in page.chars
                        if char["x0"] >= word["x0"] and char["x1"] <= word["x1"]
                    ]

                    if chars:
                        # Aggregate character-level details to word level
                        font = chars[0].get("fontname")
                        font_size = chars[0].get("size")
                        color_hex = page.annots.get("color", "#000000") if page.annots else "#000000"

                        word_styling_details.append({
                            "page": page_num + 1,
                            "text": word["text"],
                            "font": font,
                            "font_size": font_size,
                            "background_color": color_hex,
                            "bounding_box": {
                                "x0": word["x0"],
                                "y0": word["top"],
                                "x1": word["x1"],
                                "y1": word["bottom"]
                            }
                        })

        return word_styling_details
    except Exception as e:
        print(f"Error while extracting word styling: {e}")
        return []

# Example Usage
pdf_file_path = "example.pdf"
styling_data = extract_word_styling(pdf_file_path)

# Display first 10 results for demonstration
for item in styling_data[:10]:
    print(item)
