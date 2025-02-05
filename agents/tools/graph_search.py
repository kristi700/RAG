from fuzzywuzzy import fuzz
from agents.core.tool import ToolBase

class GraphSearchTool(ToolBase):
    def __init__(self, graph_db, graph_depth):
        super().__init__(
            name='graph_search',
            description='Searches the graph db for relevant information.'
        )
        self.graph_depth = graph_depth
        self.graph_db = graph_db

    def execute(self, query):
        graph_entities = self._find_related_entities(query)
        graph_context = []
        if graph_entities:
            graph_context = self._get_entity_relations(
                graph_entities,
                max_depth=self.graph_depth 
            )

        graph_texts = []
        for entry in graph_context:
            graph_texts.append(
                f"- **{entry['source']}** ({entry['source_description']})\n"
                f"  - **{entry['relationship']}** → *{entry['target']}* ({entry['target_description']})"
            )

        graph_texts = "\n".join(graph_texts)
        return graph_texts

    def _find_related_entities(self, query, max_entities: int = 5, threshold: int = 70):
        result = self.graph_db.execute_query(
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
            name_score = self._fuzzy_match(query_keywords, entity["name"])
            desc_score = self._fuzzy_match(query_keywords, entity["description"])
            if name_score > threshold or desc_score > threshold:
                relevant_entities.append({
                    'id': entity['id'],
                    'name': entity['name'],
                    'score': name_score if name_score > threshold else desc_score
                })

        return sorted(relevant_entities, key=lambda x: x['score'], reverse=True)[:max_entities]

    def _fuzzy_match(self, query_keywords, text):
        """Returns the highest fuzzy match score between the query and a given text."""
        if isinstance(text, bytes):
            text = text.decode("utf-8")

        text = text.lower()
        scores = [fuzz.partial_ratio(q, text) for q in query_keywords]
        return max(scores) if scores else 0

    def _get_entity_relations(self, entities, max_depth: int):
        relations = []
        
        for entity in entities:
            # NOTE - do we need directed rel here?
            query = (
                f"MATCH p=(e)-[r*1..{max_depth}]-(neighbor) "
                f"WHERE id(e) == {entity['id']} "
                "RETURN p;"
            )
            
            result = self.graph_db.execute_query(query)
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


    