import pdfplumber

def extract_word_background_color(pdf_path):
    try:
        word_background_details = []

        # Open the PDF
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                words = page.extract_words()
                rectangles = page.rects  # Graphical elements like filled rectangles

                for word in words:
                    word_box = {
                        "x0": word["x0"],
                        "y0": word["top"],
                        "x1": word["x1"],
                        "y1": word["bottom"],
                    }

                    # Check for rectangles overlapping with the word's bounding box
                    background_color = None
                    for rect in rectangles:
                        if (
                            rect["x0"] <= word_box["x0"] <= rect["x1"]
                            and rect["y0"] <= word_box["y0"] <= rect["y1"]
                        ):
                            background_color = rect.get("non_stroking_color")  # Background color

                    word_background_details.append({
                        "page": page_num + 1,
                        "text": word["text"],
                        "bounding_box": word_box,
                        "background_color": background_color,
                    })

        return word_background_details

    except Exception as e:
        print(f"Error while extracting word background color: {e}")
        return []

# Example Usage
pdf_file_path = "example.pdf"
background_data = extract_word_background_color(pdf_file_path)

# Display results
for item in background_data[:10]:
    print(item)
