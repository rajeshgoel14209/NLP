from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageDraw

def extract_word_colors(pdf_path):
    try:
        # Convert PDF to images (one image per page)
        images = convert_from_path(pdf_path)
        results = []

        for page_num, image in enumerate(images, start=1):
            # Use Tesseract to extract words with bounding boxes
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

            for i in range(len(data["text"])):
                word = data["text"][i].strip()
                if word:  # Only process non-empty words
                    # Extract word position
                    x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                    
                    # Extract the word region from the image
                    word_region = image.crop((x, y, x + w, y + h))
                    
                    # Get the average color of the word region
                    avg_color = word_region.resize((1, 1)).getpixel((0, 0))  # Downscale to 1x1 pixel for average color
                    
                    results.append({
                        "page": page_num,
                        "text": word,
                        "font_color": avg_color  # RGB color of the word
                    })

        return results

    except Exception as e:
        print(f"Error extracting word colors: {e}")
        return []

# Example Usage
pdf_file_path = "example.pdf"  # Replace with your PDF file path
word_color_data = extract_word_colors(pdf_file_path)

# Display results
for item in word_color_data[:10]:  # Display the first 10 words with colors
    print(item)
