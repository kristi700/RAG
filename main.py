import argparse

from utils import pdf_processor
from vector_db.vector_db import WeaviateVectorDatabase
# NOTE - add LLM choice

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
    parser.add_argument("pdf file", help="PDF file to use as context for the LLM")
    return parser.parse_known_args()

def main():
    args, modifiers = parse_args()
    vector_db = WeaviateVectorDatabase(host='host.docker.internal',port='8080')
    vector_db.delete_collection("test")
    ## testonly
    properties=[
        wvc.config.Property(
            name="question",
            data_type=wvc.config.DataType.TEXT,
        ),
        wvc.config.Property(
            name="answer",
            data_type=wvc.config.DataType.TEXT,
        ),
        wvc.config.Property(
            name="category",
            data_type=wvc.config.DataType.TEXT,
        )
    ]
    vector_db.create_collection(collection_name="test", properties=properties)

    resp = requests.get(
        "https://raw.githubusercontent.com/weaviate-tutorials/quickstart/main/data/jeopardy_tiny.json"
    )
    df = pd.DataFrame(resp.json())
    transformed_data = [{col: row[col] for col in df.columns} for _, row in df.iterrows()]

    vector_db.add_documents("test", transformed_data)
    response = vector_db.search(collection_name='test', query_vector="animal", search_type='bm25', filters = wvc.query.Filter.by_property("category").equal("Animals"))
    print(response)

if __name__ == "__main__":
    main()