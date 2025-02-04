from agents.core.tool import ToolBase
from vector_db.vector_db import WeaviateVectorDatabase

class VectorSearchTool(ToolBase):
    def __init__(self):
        super().__init__(
            name='vector_search',
            description='Searches the vector db for relevant information.'
        )

    def execute(self, collection_name: str, vector_db: WeaviateVectorDatabase, query: str, top_k: int = 3):
        result = vector_db.search(collection_name, query, top_k=top_k)
        return result


    