import json
import argparse
import weaviate.classes as wvc

from utils import preprocess
from utils.chat import chat_loop
from utils.llm_wrapper import LLM_wrapper
from graph_db.graph_db import NebulaHandler
from rag.data_ingestion import upload_to_dbs
from vector_db.vector_db import WeaviateVectorDatabase

# TODO - stg needs to be done regarding the collection name variable - useless to keep passing it back and forth!

# TODO - implement saving/caching the built graph and vector dbs so one doesnt have to recreate them all the time
# or at least save the refinmed triplets to save time on that (if pdf and llm are the same)

# NOTE - add LLM choice
# NOTE - add context file type - pdf, txt...
# NOTE - agents - like what tool to use - graph, vector or internet(?)
# NOTE - implement chat function
# NOTE - add memory to chat

def parse_args() -> argparse.Namespace:
    """
    Parse the command line arguments for RAG.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Settings for RAG")
    parser.add_argument("pdf_path", help="PDF file to use as context for the LLM")
    return parser.parse_known_args()


# TODO - make all upload func batch upload compatible - for multiple pdfs(?) - not sure i wanna implement this, some new technology would be nicer 
def main():
    args, _ = parse_args()
    collection_name = 'context_data'

    vector_db = WeaviateVectorDatabase(host='host.docker.internal',port='8080')
    graph_db = NebulaHandler(space_name=collection_name, host='host.docker.internal', port=9669)
    llm = LLM_wrapper(model_name="Qwen/Qwen2.5-0.5B-Instruct")

    extracted_text = preprocess.parse_pdf(args.pdf_path, is_sentence_split=True)
    chunked_data = preprocess.sentence_chunker(extracted_text, max_tokens=64)
    #triplets = preprocess.extract_triplets(llm, chunked_data)

    vector_db.delete_collection(collection_name) # tmp only
    # NOTE - 0.5b qwen, first run data, for testing only!
    with open("data.json") as json_file:
        triplets = json.load(json_file)
    # NOTE - multi doc based thingy wont work as of rightnow!!

    all_triplets = {'triplets':[item for json_str in triplets for item in json_str['triplets']]}
    combined_chunks = {'chunks': chunked_data}
    combined_data = [{"doc_id": 1, "text": extracted_text, 'chunks': combined_chunks['chunks'], 'triplets': all_triplets["triplets"]}] # Only one doc for now

    properties = [
        wvc.config.Property(name="name", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="description", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
    ]
    vector_db.create_collection(collection_name=collection_name, properties=properties)
    graph_db.recreate_space()

    upload_to_dbs(llm, vector_db, graph_db, collection_name, combined_data[0]) # [0] as it only works with 1 doc as of rightnow
    
    # TODO - create a multi turn chat!
    chat_loop(llm, graph_db, vector_db)

    # Cleanup
    vector_db.delete_collection(collection_name)
    vector_db.client.close()

if __name__ == "__main__":
    main()