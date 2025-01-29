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

    vector_db.add_documents(collection_name, documents_to_insert)