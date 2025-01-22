import argparse

from utils.pdf_processor import parse_pdf
from utils.chunker import sentence_chunker
from vector_db.vector_db import WeaviateVectorDatabase
# NOTE - add LLM choice
# NOTE - add context file type - pdf, txt...
import weaviate.classes as wvc
import requests
import pandas as pd

def parse_args() -> argparse.Namespace:
    """
    Parse the command line arguments for RAG.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Settings for RAG")
    parser.add_argument("pdf_path", help="PDF file to use as context for the LLM")
    return parser.parse_known_args()

def main():
    collection_name = 'context_data'
    args, modifiers = parse_args()
    extracted_text = parse_pdf(args.pdf_path, is_sentence_split=True)
    chunked_data = sentence_chunker(extracted_text, max_tokens=200)

    vector_db = WeaviateVectorDatabase(host='host.docker.internal',port='8080')
    vector_db.delete_collection(collection_name)
    ## testonly
    properties=[
        wvc.config.Property(
            name="chunk_id",
            data_type=wvc.config.DataType.INT,
        ),
        wvc.config.Property(
            name="content",
            data_type=wvc.config.DataType.TEXT,
        ),
    ]
    vector_db.create_collection(collection_name=collection_name, properties=properties)
    vector_db.add_documents(collection_name, chunked_data)

    response = vector_db.search(collection_name=collection_name, query_vector="animal", search_type='bm25')
    print(response)

if __name__ == "__main__":
    main()