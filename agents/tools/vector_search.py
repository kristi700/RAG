from agents.core.tool import ToolBase
from vector_db.vector_db import WeaviateVectorDatabase

class VectorSearchTool(ToolBase):
    def __init__(self, collection_name, vector_db, top_k):
        super().__init__(
            name='vector_search',
            description='Searches the vector db for relevant information.'
        )
        self.collection_name=collection_name
        self.vector_db = vector_db
        self.top_k = top_k

    def execute(self, query: str):
        vector_response = self.vector_db.search(
            collection_name=self.collection_name,
            query=query,
            top_k=self.top_k
        )

        vector_context = [f"Entity: {item.properties['name']},\n Description: {item.properties['description']},\n Content: {item.properties['content']}" for item in vector_response.objects]
        vector_texts = "\n".join([f"{i+1}. {text}" for i, text in enumerate(vector_context)])
        return vector_texts
        