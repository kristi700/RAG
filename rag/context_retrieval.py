from fuzzywuzzy import fuzz
from typing import List, Dict, Optional
from graph_db.graph_db import NebulaHandler
from vector_db.vector_db import WeaviateVectorDatabase

def get_context(graph_db: NebulaHandler, vector_db: WeaviateVectorDatabase, user_prompt: str, vector_top_k: int = 3,graph_depth: int = 2)-> Dict[str, List]:
    context = {"vector_context": [], "graph_context": []}
    vector_response = vector_db.search(
        collection_name='context_data',
        query=user_prompt,
        top_k=vector_top_k
    )

    vector_context = [f"Entity: {item.properties['name']},\n Description: {item.properties['description']},\n Content: {item.properties['content']}" for item in vector_response.objects]
    
    context["vector_context"] = vector_context
    graph_entities = _find_related_entities(graph_db, user_prompt)

    if graph_entities:
        context["graph_context"] = _get_entity_relations(
            graph_db,
            graph_entities,
            max_depth=graph_depth
        )

    # TODO - fix
    vector_texts = "\n".join(
        [f"{i+1}. {text}" for i, text in enumerate(context.get("vector_context", []))]
    )
    graph_texts = []
    for entry in context.get("graph_context", []):
        graph_texts.append(
            f"- **{entry['source']}** ({entry['source_description']})\n"
            f"  - **{entry['relationship']}** → *{entry['target']}* ({entry['target_description']})"
        )

    graph_texts = "\n".join(graph_texts)
    
    return {"graph_context": graph_texts, "vector_context": vector_texts}

def _find_related_entities(graph_db: NebulaHandler, query: str, max_entities: int = 5, threshold: int = 70) -> List[Dict]:
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
        name_score = _fuzzy_match(query_keywords, entity["name"])
        desc_score = _fuzzy_match(query_keywords, entity["description"])
        if name_score > threshold or desc_score > threshold:
            relevant_entities.append({
                'id': entity['id'],
                'name': entity['name'],
                'score': name_score if name_score > threshold else desc_score
            })

    return sorted(relevant_entities, key=lambda x: x['score'], reverse=True)[:max_entities]

def _fuzzy_match(query_keywords, text):
    """Returns the highest fuzzy match score between the query and a given text."""
    if isinstance(text, bytes):
        text = text.decode("utf-8")

    text = text.lower()
    scores = [fuzz.partial_ratio(q, text) for q in query_keywords]
    return max(scores) if scores else 0

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