from agents.core.tool import ToolBase

class GraphSearchTool(ToolBase):
    def __init__(self):
        super().__init__(
            name='graph_search',
            description='Searches the graph db for relevant information.'
        )

    def execute(self):
        ...


    