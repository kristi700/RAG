import re
import json
import nltk
import unicodedata
from typing import List
from pypdf import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer
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

def extract_entity_types(chunked_data, spacy_nlp):
    """Extract unique entity types from the text using NER."""
    entity_types = set()
    for data in chunked_data:
        doc = spacy_nlp(data['content'])
        entity_types.update(ent.label_ for ent in doc.ents)
    return list(entity_types)

def extract_predicates(chunked_data, spacy_nlp):
    """
    Extract potential predicates (verbs) from the text using dependency parsing.
    """
    predicates = set()
    for data in chunked_data:
        doc = spacy_nlp(data['content'])
        for token in doc:
            if token.pos_ == "VERB" or token.dep_ == "ROOT":
                predicates.add(token.text)
    return list(predicates)

def triplextract(model, tokenizer, text, entity_types, predicates):

    input_format = """Perform Named Entity Recognition (NER) and extract knowledge graph triplets from the text. NER identifies named entities of given entity types, and triple extraction identifies relationships between entities using specified predicates.
      
        **Entity Types:**
        {entity_types}
        
        **Predicates:**
        {predicates}
        
        **Text:**
        {text}
        """

    message = input_format.format(
                entity_types = json.dumps({"entity_types": entity_types}),
                predicates = json.dumps({"predicates": predicates}),
                text = text)

    messages = [{'role': 'user', 'content': message}]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt = True, return_tensors="pt")
    output = tokenizer.decode(model.generate(input_ids=input_ids, max_length=256)[0], skip_special_tokens=True)
    return output

def init_triplex():
    model = AutoModelForCausalLM.from_pretrained("sciphi/triplex", trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained("sciphi/triplex", trust_remote_code=True)
    return model, tokenizer