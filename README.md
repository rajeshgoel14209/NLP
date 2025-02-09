import base64

def encode_pdf_to_base64(pdf_path, output_base64_path):
    with open(pdf_path, "rb") as pdf_file:
        encoded_string = base64.b64encode(pdf_file.read()).decode('utf-8')
    
    with open(output_base64_path, "w") as base64_file:
        base64_file.write(encoded_string)

    print(f"Encoded Base64 saved to {output_base64_path}")


    def decode_base64_to_pdf(base64_path, output_pdf_path):
    with open(base64_path, "r") as base64_file:
        encoded_string = base64_file.read()
    
    with open(output_pdf_path, "wb") as pdf_file:
        pdf_file.write(base64.b64decode(encoded_string))

    print(f"Decoded PDF saved to {output_pdf_path}")
