import spacy
import argparse
import utils.preprocess as preprocess

from typing import Optional
from utils.llm_wrapper import LLM_wrapper
from graph_db.graph_db import NebulaHandler
from vector_db.vector_db import WeaviateVectorDatabase

from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

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
 

def insert_data_to_graphdb(graph_db, collection_name, all_docs):
    graph_db.switch_space(space_name=collection_name)
    for doc in all_docs:
        doc_triplets = doc["triplets"]

        for trip in doc_triplets:
            subject = trip["subject"]
            predicate = trip["predicate"]
            obj = trip["object"]

            subject_name = subject["name"]
            subject_desc = subject.get("description", "")
            
            object_name = obj["name"]
            object_desc = obj.get("description", "")

            graph_db.upsert_entity_relationship(
                src_name=subject_name,
                src_description=subject_desc,
                dst_name=object_name,
                dst_description=object_desc,
                relationship=predicate
            )

def insert_data_to_vectordb(vector_db, collection_name, all_docs):
    documents_to_insert = []
    for doc in all_docs:
        doc_id = doc["doc_id"]
        for chunk in doc["chunks"]:
            documents_to_insert.append({
                "doc_id": doc_id,
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"]
            })

        for trip in doc["triplets"]:
            subject = trip["subject"]
            obj = trip["object"]

            documents_to_insert.append({
                "doc_id": doc_id,
                "chunk_id": f"triplet-{trip['predicate']}",
                "content": f"{subject['name']} ({subject.get('description', '')}) {trip['predicate']} {obj['name']} ({obj.get('description', '')})"
            })

    vector_db.add_documents(collection_name, documents_to_insert)


## NOTE - not sure of this docid thingy liek this nglxd

def get_context(graph_db: NebulaHandler, vector_db, user_prompt: str,doc_id: Optional[int] = None, vector_top_k: int = 3,graph_depth: int = 2)-> Dict[str, List]:
    context = {"vector_context": [], "graph_context": []}
    vector_response = vector_db.search(
        collection_name='context_data',
        query=user_prompt,
        top_k=vector_top_k
    )

    if doc_id is not None:
        vector_context = [
            item.properties['content'] for item in vector_response.objects
            if item.properties.get('doc_id') == doc_id
        ]
    else:
        vector_context = [item.properties['content'] for item in vector_response.objects]
    
    context["vector_context"] = vector_context
    graph_entities = _find_related_entities(graph_db, user_prompt)

    if graph_entities:
        context["graph_context"] = _get_entity_relations(
            graph_db,
            graph_entities,
            max_depth=graph_depth
        )
    
    return context

def _find_related_entities(graph_db: NebulaHandler, query: str, max_entities: int = 5) -> List[Dict]:
    result = graph_db.execute_query(
        "LOOKUP ON entity YIELD id(vertex) AS vid, properties(vertex).name AS name, "
        "properties(vertex).description AS description"
    )

    if not result or not result.rows():
        return []

    entities = []
    for row in result.rows():
        entities.append({
            "id": row.values[0].get_iVal(),
            "name": row.values[1].get_sVal(),
            "description": row.values[2].get_sVal()
        })

    query_keywords = set(query.lower().split())
    relevant_entities = []
    

    # NOTE - do we only get entity name?
    for entity in entities:
        desc = entity.get('description', '').lower().decode()
        score = len(query_keywords.intersection(set(desc.split())))
        if score > 0:
            relevant_entities.append({
                'id': entity['id'],
                'name': entity['name'],
                'score': score
            })

    return sorted(relevant_entities, key=lambda x: x['score'], reverse=True)[:max_entities]


def _get_entity_relations(graph_db: NebulaHandler, entities: List[Dict], max_depth: int = 2) -> List[Dict]:
    relations = []
    
    for entity in entities:
        # NOTE - do we need directed rel here?
        query = (
            f"MATCH p=(e)-[r*1..{max_depth}]-(neighbor) "
            f"WHERE id(e) == {entity['id']} "
            "RETURN p;"
        )
        
        result = graph_db.execute_query(query)
        if not result or not result.rows():
            continue

        for row in result.rows():
            path = row.values[0].value
            src_vertex = path.src
            source = src_vertex.tags[0].props[b'name'].value.decode()
            source_description = src_vertex.tags[0].props.get(b'description', b'').value.decode()

            for step in path.steps:
                dst_vertex = step.dst
                target = dst_vertex.tags[0].props[b'name'].value.decode()
                target_description = dst_vertex.tags[0].props.get(b'description', b'').value.decode()
                relationship = step.props[b'relationship'].value.decode()

                relations.append({
                    'source': source,
                    'source_description': source_description,
                    'relationship': relationship,
                    'target': target,
                    'target_description': target_description,
                })
    
    return relations

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

    context = get_context(graph_db, vector_db, args.question)
    print(llm.generate(user_prompt=args.question, context=context))
    vector_db.client.close()
if __name__ == "__main__":
    main()