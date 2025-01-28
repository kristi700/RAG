import spacy
import argparse
import utils.preprocess as preprocess

from typing import Optional
from utils.llm_wrapper import LLM_wrapper
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
    parser.add_argument("question", help="User question to answer with context")
    parser.add_argument("pdf_path", help="PDF file to use as context for the LLM")
    return parser.parse_known_args()
 

def insert_data_to_graphdb(graph_db, collection_name, all_docs):
    graph_db.switch_space(space_name=collection_name)
    for doc in all_docs:
        doc_triplets = doc["triplets"]
        doc_entities = doc.get("entities", [])

        entity_map = {}
        for ent in doc_entities:
            entity_map[ent["name"]] = f"Type: {ent['type']}"

        for trip in doc_triplets:
            subject_name = trip["subject"]
            predicate = trip["predicate"]
            object_name = trip["object"]

            subject_desc = entity_map.get(subject_name, "")
            object_desc = entity_map.get(object_name, "")

            graph_db.upsert_entity_relationship(
                src_name=subject_name,
                src_description=subject_desc,
                dst_name=object_name,
                dst_description=object_desc,
                relationship=predicate
            )

def insert_data_to_vectordb(vector_db, collection_name, all_docs):
    documents_to_insert =[]
    for doc in all_docs:
        doc_id = doc["doc_id"]
        for chunk in doc["chunks"]:
            documents_to_insert.append({
                "doc_id": doc_id,
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"]
            })

    vector_db.add_documents(collection_name, documents_to_insert)


## NOTE - not sure of this docid thingy liek this nglxd
def get_context(graph_db, vector_db, user_prompt: str, doc_id: Optional[int] = None):
    # TODO -review the grpah db!
    weaviate_response = vector_db.search('context_data', user_prompt, top_k=2) # dont hardcode!
    vector_context = [item.properties['content'] for item in weaviate_response.objects if item.properties['doc_id'] == doc_id]
    #graph_results = graph_db.
    return vector_context

# TODO - make all upload func batch upload compatible - for multiple pdfs!
# TODO - make the format of the data the same as in the dummy_data.json (combine chunked + raw + triplets)
def main():
    collection_name = 'context_data'
    args, modifiers = parse_args()
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

    import json
    with open("dummy_data.json") as json_file:
        combined_data = json.load(json_file)
    """
    prediction = []
    for chunk in chunked_data:
        prediction.append(preprocess.triplextract(triplex, triplex_token, chunk, entity_types, predicates))

    combined_data = TODO
    """
    vector_db.delete_collection(collection_name)
    ## testonly
    properties = [
        wvc.config.Property(
            name="doc_id",
            data_type=wvc.config.DataType.INT,
        ),
        wvc.config.Property(
            name="chunk_id",
            data_type=wvc.config.DataType.INT,
        ),
        wvc.config.Property(
            name="content",
            data_type=wvc.config.DataType.TEXT,
        ),
    ]
    
    vector_db.create_collection(
        collection_name=collection_name,
        properties=properties
    )
    graph_db.recreate_space()

    insert_data_to_vectordb(vector_db, collection_name, combined_data)
    insert_data_to_graphdb(graph_db, collection_name, combined_data)

    context = get_context(graph_db, vector_db, args.question, doc_id=1)
    print(llm.generate(user_prompt=args.question, context=context))
    vector_db.client.close()
if __name__ == "__main__":
    main()