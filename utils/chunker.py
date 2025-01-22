import nltk
nltk.download("punkt_tab")

def sentence_chunker(text, max_tokens: int):
    sentences = nltk.tokenize.sent_tokenize(text, language='english')
    chunks = []
    current_chunk = []
    current_tokens = 0
    chunk_id = 1
    
    for sentence in sentences:
        token_count = len(nltk.tokenize.word_tokenize(sentence))
        if current_tokens + token_count > max_tokens:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "chunk_id": chunk_id,
                "content": chunk_text
            })
            chunk_id += 1
            current_chunk = [sentence]
            current_tokens = token_count
        else:
            current_chunk.append(sentence)
            current_tokens += token_count

    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append({
            "chunk_id": chunk_id,
            "content": chunk_text
        })
    
    return chunks