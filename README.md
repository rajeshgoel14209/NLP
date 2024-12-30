def split_text_into_chunks(text, chunk_size=100):
    """
    Splits a text into chunks of specified size if the length exceeds the chunk size.

    Args:
        text (str): The input text to be split.
        chunk_size (int): The maximum size of each chunk. Default is 100.

    Yields:
        str: Chunks of the text.
    """
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        yield text[start:end]
        start = end

# Example usage
text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " \
       "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. " \
       "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris " \
       "nisi ut aliquip ex ea commodo consequat."

if len(text) > 100:
    print("Text length exceeds 100. Splitting into chunks:")
    for chunk in split_text_into_chunks(text):
        print(chunk)
else:
    print("Text length is within the limit. No splitting required.")
