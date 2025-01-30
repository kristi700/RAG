import json
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

    vector_db = WeaviateVectorDatabase(host='host.docker.internal',port='8080')
    graph_db = NebulaHandler(space_name=collection_name, host='host.docker.internal', port=9669)
    llm = LLM_wrapper(model_name="Qwen/Qwen2.5-0.5B-Instruct")

    #triplex, triplex_token = preprocess.init_triplex()
    extracted_text = preprocess.parse_pdf(args.pdf_path, is_sentence_split=True)
    chunked_data = preprocess.sentence_chunker(extracted_text, max_tokens=64)
    # would be nice to tqdm this ngl
    #triplets = preprocess.extract_triplets(llm, chunked_data)

    vector_db.delete_collection(collection_name)
    # NOTE -0.5b qwen, first runs data, for testing only!
    with open("combined_triplets.json") as json_file:
        triplets = json.load(json_file)

    """
    # NOTE - multi doc based thingy wont work as of rightnow!!

    # as entities are not needed, all we need to do is
    #   - 
    #   - combine that with chunked data into one list

    all_triplets = {'triplets': []}

    for triplet in triplets:
        all_triplets['triplets'] += json.loads(triplet)['triplets']
    combined_data = TODO
    """

    #tmptmptmp
    combined_chunks = {'chunks': []}
    for i in range(len(chunked_data)):
        combined_chunks['chunks'].append(chunked_data[i])

    triplets = triplets # refine triplets (and making sure all is in the same dict)

    # Only one doc for now
    combined_data = [{"doc_id": 1, "text": extracted_text, 'chunks': combined_chunks['chunks'], 'triplets': triplets['triplets']}]

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
    
    print(llm.generate_chat(user_prompt=args.question, context=context))

    # Cleanup
    vector_db.delete_collection(collection_name)
    vector_db.client.close()

if __name__ == "__main__":
    main()