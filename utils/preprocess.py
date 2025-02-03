import re
import json
import nltk
import logging
import unicodedata

from tqdm import tqdm
from typing import List
from pypdf import PdfReader
from json_repair import repair_json
from utils.llm_wrapper import LLM_wrapper
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
        
        if current_tokens + token_count > max_tokens and current_chunk:
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

def extract_triplets(llm: LLM_wrapper, chunked_data):
    combined_triplets = []
    failed_chunks = []
    
    for chunk_id, chunk in tqdm(enumerate(chunked_data), total=len(chunked_data), desc='Triplet extraction'):
        try:
            raw_triplet_text = llm.generate_extract(chunk["content"])
            parsed_triplets = repair_json(raw_triplet_text)
            as_dictionary = json.loads(parsed_triplets)
            for triplet in as_dictionary['triplets']:
                triplet['chunk_id'] = chunk_id

            if as_dictionary and "triplets" in as_dictionary:
                combined_triplets.append(str(as_dictionary))
            else:
                logging.warning(f"Skipping chunk due to missing 'triplets': {raw_triplet_text}")
                failed_chunks.append(chunk["content"])

        except Exception as e:
            logging.error(f"Error processing chunk: {chunk['content'][:50]}... - {e}")
            failed_chunks.append(chunk["content"])

    return combined_triplets, failed_chunks