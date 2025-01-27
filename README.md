import pytesseract
import cv2
from PIL import Image
import numpy as np

# Configure pytesseract path if not added to PATH (Windows only)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_table_from_image(image_path):
    # Step 1: Load the image
    img = cv2.imread(image_path)

    # Step 2: Preprocess the image for OCR
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, binary_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY_INV)  # Threshold to binary

    # Step 3: Detect table structure (optional - can be fine-tuned based on your image)
    kernel = np.ones((5, 5), np.uint8)  # Define kernel for morphological operations
    dilated_img = cv2.dilate(binary_img, kernel, iterations=2)  # Enhance table lines

    # Step 4: Use pytesseract to extract the text while preserving layout
    custom_config = r'--oem 3 --psm 6'  # Page segmentation mode 6 (assumes a single uniform block of text)
    extracted_text = pytesseract.image_to_string(dilated_img, config=custom_config)

    return extracted_text

# Example usage
image_path = "table_image.png"  # Replace with your table image path
table_data = extract_table_from_image(image_path)

# Print extracted text from the table
print(table_data)
