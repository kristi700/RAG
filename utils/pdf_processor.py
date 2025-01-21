import re
import unicodedata
from typing import List
from pypdf import PdfReader


def clean_text(text: str, lowercase: bool = True) -> str:
    """
    Cleans a text string by:
      -Normalizing unicode
      -Lowercasing
      -Removing punctuation and special characters
      -Removing extra whitespace
    """

    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("utf-8", "ignore")
    if lowercase:
        text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())

    return text


def parse_pdf(document_path: str) ->List[str]:
    """
    Extracts pure text from the provided .pdf file and removes special charachters.
    """
    reader = PdfReader(document_path)
    pure_text = []
    for i in range(len(reader.pages)):
        extracted_text=reader.pages[i].extract_text()
        cleaned = clean_text(extracted_text)
        pure_text.append(cleaned)

    return pure_text