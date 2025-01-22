import argparse
import utils.preprocess as preprocess

from graph_db.graph_db import NebulaHandler
from vector_db.vector_db import WeaviateVectorDatabase

# NOTE - add LLM choice
# NOTE - add context file type - pdf, txt...
import weaviate.classes as wvc

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
    vector_db = WeaviateVectorDatabase(host='host.docker.internal',port='8080')
    #graph_db = NebulaHandler(space_name=collection_name, host='host.docker.internal', port=9669)
    ner_pipeline = preprocess.init_ner_pipiline()

    extracted_text = preprocess.parse_pdf(args.pdf_path, is_sentence_split=True)
    chunked_data = preprocess.sentence_chunker(extracted_text, max_tokens=64)

    # would be nice to tqdm it somehow ngl
    ner_output = ner_pipeline(chunked_data)
    from transformers import pipeline
    triplet_extractor = pipeline(
    "text2text-generation",
    model="SciPhi/Triplex",
    tokenizer="SciPhi/Triplex"
    )
    output = triplet_extractor(
    ner_output,
    max_length=64,
    do_sample=False
    )
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