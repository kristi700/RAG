
import json
import spacy
import argparse
import weaviate.classes as wvc

from utils import preprocess
from utils.llm_wrapper import LLM_wrapper
from graph_db.graph_db import NebulaHandler
from rag.context_retrieval import get_context
from vector_db.vector_db import WeaviateVectorDatabase
from rag.data_ingestion import insert_data_to_graphdb, insert_data_to_vectordb

# NOTE - add LLM choice
# NOTE - add context file type - pdf, txt...
# TODO - add graph triplet optimization!

import weaviate.classes as wvc

def parse_args() -> argparse.Namespace:
    """
    Parse the command line arguments for RAG.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Settings for RAG")
    parser.add_argument("question", help="User question to answer with context")
    parser.add_argument("pdf_path", help="PDF file to use as context for the LLM")
    return parser.parse_known_args()


# TODO - make all upload func batch upload compatible - for multiple pdfs!
# TODO - make the format of the data the same as in the dummy_data.json (combine chunked + raw + triplets)
def main():
    args, _ = parse_args()
    collection_name = 'context_data'

    spacy_nlp = spacy.load("en_core_web_sm")
    vector_db = WeaviateVectorDatabase(host='host.docker.internal',port='8080')
    graph_db = NebulaHandler(space_name=collection_name, host='host.docker.internal', port=9669)
    llm = LLM_wrapper(model_name="Qwen/Qwen2.5-0.5B-Instruct")

    #triplex, triplex_token = preprocess.init_triplex()
    extracted_text = preprocess.parse_pdf(args.pdf_path, is_sentence_split=True)
    chunked_data = preprocess.sentence_chunker(extracted_text, max_tokens=64)

    # TODO - should review and optimize this!
    entity_types = preprocess.extract_entity_types(chunked_data, spacy_nlp)
    predicates = preprocess.extract_predicates(chunked_data, spacy_nlp)
    # would be nice to tqdm it somehow ngl

    with open("dummy_data.json") as json_file:
        combined_data = json.load(json_file)
    """
    prediction = []
    for chunk in chunked_data:
        prediction.append(preprocess.triplextract(triplex, triplex_token, chunk, entity_types, predicates))

    combined_data = TODO
    """
    ## testonly
    properties = [
        wvc.config.Property(name="doc_id", data_type=wvc.config.DataType.INT),
        wvc.config.Property(name="chunk_id", data_type=wvc.config.DataType.INT),
        wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
    ]
    vector_db.create_collection(collection_name=collection_name, properties=properties)
    graph_db.recreate_space()

    insert_data_to_vectordb(vector_db, collection_name, combined_data)
    insert_data_to_graphdb(graph_db, collection_name, combined_data)

    context = get_context(graph_db, vector_db, args.question)
    
    print(llm.generate(user_prompt=args.question, context=context))

    # Cleanup
    vector_db.delete_collection(collection_name)
    vector_db.client.close()

if __name__ == "__main__":
    main()