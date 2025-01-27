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


# TODO - make all upload func batch upload compatible - for multiple pdfs!
# TODO - make the format of the data the same as in the dummy_data.json (combine chunked + raw + triplets)
def main():
    collection_name = 'context_data'
    args, modifiers = parse_args()
    vector_db = WeaviateVectorDatabase(host='host.docker.internal',port='8080')
    graph_db = NebulaHandler(space_name=collection_name, host='host.docker.internal', port=9669)
    #triplex, triplex_token = preprocess.init_triplex()
    extracted_text = preprocess.parse_pdf(args.pdf_path, is_sentence_split=True)
    chunked_data = preprocess.sentence_chunker(extracted_text, max_tokens=64)
    entity_types = [ "LOCATION", "POSITION", "DATE", "CITY", "COUNTRY", "NUMBER" ]
    predicates = [ "POPULATION", "AREA" ]
    # would be nice to tqdm it somehow ngl

    # TODO - make the data format as of the predictions
    import json
    with open("dummy_data.json") as json_file:
        combined_data = json.load(json_file)
    """
    prediction = []
    for chunk in chunked_data:
        prediction.append(preprocess.triplextract(triplex, triplex_token, chunk, entity_types, predicates))
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

    response = vector_db.search(collection_name=collection_name, query_vector="animal", search_type='bm25')
    print(response)

if __name__ == "__main__":
    main()