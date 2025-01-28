def sliding_window_tokenizer(text, max_length=512, stride=128):
    tokens = tokenizer(text, truncation=False)["input_ids"]
    chunks = []
    for i in range(0, len(tokens), max_length - stride):
        chunks.append(tokens[i:i + max_length])
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
