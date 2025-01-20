import syslog
import weaviate
import weaviate.classes as wvc

from tqdm import tqdm
from typing import List, Dict, Optional
from weaviate.classes.query import Filter
class WeaviateVectorDatabase:
    def __init__(self, host: str, port: int):
        """
        Initialize the Weaviate client.
        """
        self.client = weaviate.connect_to_local(host=host, port=port)

    def create_collection(self, collection_name: str, properties: List[Dict], vectorizer: str = "text2vec-transformers"):
        """
        Create a collection (class) in Weaviate.
        """
        self._assert_collection_not_exists()

        if vectorizer == "text2vec-transformers":
            _ = self.client.collections.create(
                name=collection_name,
                vectorizer_config=[wvc.config.Configure.NamedVectors.text2vec_transformers(name="vectorzier")],
                properties=properties
            )
            syslog.syslog(f"Collection {collection_name} created.")
        else:
            NotImplementedError("Vectorizer not implemented.")

    def delete_collection(self, name: str):
        """
        Delete a collection (class) in Weaviate.
        """
        self.client.collections.delete(name)
        syslog.syslog(f"Collection {name} deleted.")

    def _get_collection_names(self) -> List[str]:
        """
        List all collections (classes) in Weaviate.
        """
        response = self.client.collections.list_all(simple=True)
        return response

    def add_documents(self, collection_name: str, documents: List[Dict], embeddings: Optional[List[List[float]]] = None):
        """
        Add documents with optional embeddings.
        """
        collection = self.client.collections.get(collection_name)

        with collection.batch.dynamic() as batch:
            for i, document in tqdm(enumerate(documents)):
                if embeddings:
                    batch.add_object(properties = document, vector=embeddings[i])
                else:
                    batch.add_object(properties = document)

        if len(collection.batch.failed_objects) > 0:
            syslog.syslog(f"Failed to import {len(collection.batch.failed_objects)} objects")

    def search(self, collection_name: str, query_vector: List[float], top_k: int = 5, search_type:str='semantic', filters: Optional[Dict] = None) -> List[Dict]:
        """
        Search the collection using a query vector with optional filters.
        """
        self._assert_collection_exists(collection_name)

        collection = self.client.collections.get(collection_name)
        if str.lower(search_type) == 'semantic':
            return collection.query.near_text(query=query_vector, limit=top_k, filters=filters)
        elif str.lower(search_type) == 'bm25':
            return collection.query.bm25(query=query_vector, limit=top_k, filters=filters)
        else:
            NotImplementedError("Search type not implemented. Try: Semantic or BM25")


    def delete_documents(self, collection_name: str, weaviate_filter: Optional[Filter]=None, document_ids: Optional[List[str]]=None):
        """
        Delete documents by their IDs.
        """
        self._assert_collection_exists(collection_name)
        collection = self.client.collections.get(collection_name)
        if document_ids:
            collection.data.delete_many(
                where=Filter.by_id().contains_any(document_ids)
            )
        elif filter:
            collection.data.delete_many(
                where= weaviate_filter
            )
        else:
            syslog.syslog("No filter nor id were provided! No deletion has been done.")


    def _assert_collection_not_exists(self, collection_name: str):
        existing_collections = self._get_collection_names()
        existing_collection_names = [collection_name.lower() for collection_name in existing_collections.keys()]
        if str.lower(collection_name) in existing_collection_names:
            raise ValueError(f"Collection '{collection_name}' already exists. Please choose a different name.")
        
    def _assert_collection_exists(self, collection_name: str):
        existing_collections = self._get_collection_names()
        existing_collection_names = [collection_name.lower() for collection_name in existing_collections.keys()]
        if str.lower(collection_name) not in existing_collection_names:
            raise ValueError(f"Collection '{collection_name}' does not exist!")
