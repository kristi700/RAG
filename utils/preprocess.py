import re
import nltk
import unicodedata
from typing import List
from pypdf import PdfReader
from transformers import pipeline

nltk.download("punkt_tab")

def clean_text(text: str, is_lowercase: bool = True, is_punctation: bool = True) -> str:
    """
    Cleans a text string by:
      -Normalizing unicode
      -Lowercasing
      -Removing punctuation and special characters
      -Removing extra whitespace
    """

    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("utf-8", "ignore")
    if is_lowercase:
        text = text.lower()
    if not is_punctation:
        text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())

    return text


def parse_pdf(document_path: str, is_sentence_split: bool) ->List[str]:
    """
    Extracts pure text from the provided .pdf file and removes special charachters.
    """
    reader = PdfReader(document_path)
    cleaned_pages = []
    for i in range(len(reader.pages)):
        extracted_text=reader.pages[i].extract_text()
        cleaned_text = clean_text(extracted_text, not is_sentence_split)
        cleaned_pages.append(cleaned_text)

    return " ".join(cleaned_pages)


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
            chunks.append(chunk_text)
            chunk_id += 1
            current_chunk = [sentence]
            current_tokens = token_count
        else:
            current_chunk.append(sentence)
            current_tokens += token_count

    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(chunk_text)
    
    return chunks

def init_ner_pipiline(model: str = 'dslim/bert-base-NER'):
    # TODO - do some checkings later
    return pipeline("ner", model=model)