import fitz  # PyMuPDF

def extract_text_and_color(pdf_path):
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text_instances = page.get_text("dict")["blocks"]
            for block in text_instances:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"]
                            color = span["color"]  # Color value
                            color_rgb = fitz.Colorspace.convert_to_rgb(color)  # RGB tuple
                            print(f"Page {page_num}: '{text}' - RGB: {color_rgb}")

# Example Usage
extract_text_and_color("example.pdf")
