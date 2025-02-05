import nltk
import syslog

from tqdm import tqdm
from random import shuffle
from utils.llm_wrapper import LLM_wrapper
from graph_db.graph_db import NebulaHandler
from vector_db.vector_db import WeaviateVectorDatabase

def is_valid_triplet(triplet):
    # NOTE - restriction to have only these keys could be useful(?)
    required_keys = {"subject", "predicate", "object"}
    if not all(key in triplet for key in required_keys):
        return False
    if not isinstance(triplet["subject"], dict) or not isinstance(triplet["object"], dict):
        return False
    if "description" not in triplet["subject"] or "description" not in triplet["object"]:
        return False
    if "name" not in triplet["subject"] or "name" not in triplet["object"]:
        return False

    return True

def find_similar_graph_nodes(collection_name: str, graph_db: NebulaHandler, vector_db: WeaviateVectorDatabase, entity: str, edit_distance_threshold=2, max_edit_distance_based=4, top_k=3):
    all_node_names = graph_db.get_all_node_names()
    close_matches = []
    for similar_name in all_node_names:
        if nltk.edit_distance(similar_name, entity["name"]) <= edit_distance_threshold: # Levenshtein distance
            close_matches.append(similar_name)

    shuffle(close_matches)
    if len(close_matches) > max_edit_distance_based:
        close_matches = close_matches[:max_edit_distance_based]

    result = vector_db.search(collection_name, entity['description'], top_k=top_k)
    for obj in result.objects:
        close_matches.append(obj.properties['name'])

    return close_matches


def upload_to_dbs(llm: LLM_wrapper, vector_db: WeaviateVectorDatabase, graph_db: NebulaHandler, collection_name: str, combined_data):
    for triplet in tqdm(combined_data['triplets'], desc="Triplet refinement and DB upload"):
        try:
            if not is_valid_triplet(triplet):
                #print(f"Invalid triplet: {triplet}")
                continue

            existing_subject_nodes = find_similar_graph_nodes(collection_name, graph_db, vector_db, triplet['subject'])
            existing_object_nodes  = find_similar_graph_nodes(collection_name, graph_db, vector_db, triplet['object'])

            similars = existing_subject_nodes + existing_object_nodes
            # NOTE - calling get_description_by_name 2x might not be the most efficient way of doing this(?)
            valid_nodes = [
                (s, graph_db.get_description_by_name(s)) 
                for s in similars
                if graph_db.get_description_by_name(s) is not None
            ]
            similars, descriptions = zip(*valid_nodes) if valid_nodes else ([], [])

            data = {}
            data["document"] = combined_data['text']
            data["triplet"] = {"subject": triplet['subject']['name'], "predicate": triplet['predicate'], "object": triplet['object']['name']}
            data["new_nodes"] = [{"entity": triplet['subject']['name'], "description": triplet['subject']['description']}, {"entity": triplet['object']['name'], "description": triplet['object']['description']}]
            data["existing_nodes"] = [{"entity": similar, "description": description} for similar, description in zip(similars, descriptions)]
            """
            response = llm.refine_triplet(data)
            repaired_response = repair_json(response)
            json_response = json.loads(repaired_response)
            """
            json_response = {'refined_triplet': triplet}
            final_triplet = {
                "subject": json_response['refined_triplet']["subject"]["name"],
                "predicate": json_response['refined_triplet']["predicate"],
                "object": json_response['refined_triplet']["object"]["name"]
            }
            final_descriptions = {
                json_response['refined_triplet']["subject"]["name"]: json_response['refined_triplet']["subject"]["description"],
                json_response['refined_triplet']["object"]["name"]: json_response['refined_triplet']["object"]["description"]
            }

            # TODO - do we need this? - use json_repair(?)
            subj = final_triplet["subject"].replace("'", " ")
            obj = final_triplet["object"].replace("'", " ")
            pred = final_triplet["predicate"].replace("'", " ")
            subj_desc = final_descriptions[final_triplet["subject"]].replace("'", " ")
            obj_desc = final_descriptions[final_triplet["object"]].replace("'", " ")

            subj_vid, obj_vid = graph_db.upsert_entity_relationship(subj, subj_desc, obj, obj_desc, pred)

            # TODO - try with refined triplets as well!! - it only from data.json now!
            vector_db.upsert_entity(collection_name, subj, subj_desc, subj_vid, combined_data['chunks'][triplet['chunk_id']]['content'])
            vector_db.upsert_entity(collection_name, obj, obj_desc, obj_vid, combined_data['chunks'][triplet['chunk_id']]['content'])

        except Exception as e:
            syslog.syslog(f"Exception {e} occured.") # NOTE - stg more refined could be nice(!)
            continue

    print("Upload to databases completed successfully.")
