import olmocr
import fitz  # PyMuPDF for extracting images
import re
import spacy
from pypdf import PdfReader

# Load NLP model for text processing
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF using olmOCR, falling back to OCR if needed."""
    reader = PdfReader(pdf_path)
    full_text = ""

    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text = page.extract_text()

        # If text extraction fails, use OCR
        if not text.strip():
            print(f"Using OCR on page {page_num + 1}")
            images = extract_images_from_pdf(pdf_path, page_num)
            for img in images:
                text += olmocr.ocr(img)

        full_text += text + "\n\n"

    return full_text

def extract_images_from_pdf(pdf_path, page_num):
    """Extract images from a specific PDF page using PyMuPDF."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    images = []

    for img in page.get_images(full=True):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        images.append(image_bytes)

    return images

def identify_headers_and_sections(text):
    """Identify headers, sections, and subsections using NLP and regex."""
    headers = []
    structured_content = {}

    # Split text into lines
    lines = text.split("\n")

    for line in lines:
        line = line.strip()

        # Identify headers using regex (example: "1. Introduction" or "Section 2.3")
        if re.match(r'^\d+(\.\d+)*\s+[A-Za-z]', line):
            headers.append(line)
            structured_content[line] = []
        elif headers:
            structured_content[headers[-1]].append(line)

    return structured_content

# Example usage
if __name__ == "__main__":
    pdf_file = "sample.pdf"  # Replace with your PDF file path
    extracted_text = extract_text_from_pdf(pdf_file)
    
    structured_data = identify_headers_and_sections(extracted_text)

    # Save structured data
    with open("structured_output.txt", "w", encoding="utf-8") as f:
        for header, content in structured_data.items():
            f.write(header + "\n")
            f.write("\n".join(content) + "\n\n")

    print("Structured extraction complete! Saved to structured_output.txt.")



############################################################################################################################


from transformers import AutoProcessor, Blip2ForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
import torch
from PIL import Image

# Load BLIP-2 Vision Encoder
blip_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

# Load Mistral-7B Model
mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct")
mistral_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct")

# Process Image
image = Image.open("image.jpg")
inputs = blip_processor(image, return_tensors="pt")

# Generate image description
blip_output = blip_model.generate(**inputs)
image_caption = blip_processor.batch_decode(blip_output, skip_special_tokens=True)[0]

# Use Mistral to answer based on the image caption
query = f"Describe this image in detail: {image_caption}"
inputs = mistral_tokenizer(query, return_tensors="pt")
output = mistral_model.generate(**inputs)

response = mistral_tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
