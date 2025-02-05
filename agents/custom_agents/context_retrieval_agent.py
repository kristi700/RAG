import json
import syslog

from json_repair import repair_json
from agents.core.agent import AgentBase
from utils.llm_wrapper import LLM_wrapper
from agents.tools.graph_search import GraphSearchTool
from agents.tools.vector_search import VectorSearchTool

# TODO - refine!
SYSTEM_PROMPT = """
You are the Context Retrieval Agent.

Your task is to retrieve the most relevant information from available knowledge sources to answer user queries. You have access to:
- A **Vector Database (VectorDB)** for semantic text search.
- A **Graph Database (GraphDB)** for structured knowledge and relationships.

**Guidelines for Using the Tools:**
1. **Use the VectorDB tool** when the query involves **finding information, documents, or general knowledge** (e.g., "What is quantum entanglement?").
2. **Use the GraphDB tool** when the query requires **retrieving relationships, connections, or structured knowledge** (e.g., "Who collaborated with Dr. Smith?").
3. If a query could benefit from both sources, retrieve data from **both** tools and merge results.
4. If no relevant results are found, return "No relevant information retrieved from available sources."

**Answer Formatting:**
- Clearly summarize retrieved information.
- Combine results from multiple sources when applicable.
- If the query is vague, suggest a refinement.
"""

class ContextRetrievalAgent(AgentBase):
    def __init__(self, llm: LLM_wrapper, vector_tool: VectorSearchTool, graph_tool: GraphSearchTool):
        super().__init__(
            name="Context Retrieval Agent",
            llm=llm,
            tools={"vector_db": vector_tool, "graph_db": graph_tool},
            system_prompt=SYSTEM_PROMPT)
        
    def run(self, query: str) -> str:
        syslog.syslog(f"{self.name}: Starting Context Retrieval")
        selected_tools = self._decide_tools(query)
        syslog.syslog(f"{self.name}: {selected_tools} have been selected based on the user query")
        tool_results = self._execute_tools(query, selected_tools)
        return tool_results
    
    def _decide_tools(self, query: str) -> list:
        """Uses the LLM to choose tools based on the query."""
        prompt = f"""
        {SYSTEM_PROMPT}

        User Query: {query}

        Decide which tool(s) to use. Respond with JSON:
        {{
            "tools": ["vector_db", "graph_db"],
            "reason": "your_reason_here"
        }}
        """
        response = self.llm.vanilla_generate(prompt)
        try:
            response = repair_json(response)
            decision = json.loads(response)
            return decision["tools"]
        except:
            return ["vector_db", "graph_db"]
    
    def _execute_tools(self, query: str, tools):
        """Runs the selected tools and returns their results."""
        results = {}
        for tool_name in tools:
            if tool_name in self.tools:
                results[tool_name] = self.tools[tool_name].execute(query)
        return results