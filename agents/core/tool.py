from abc import ABC, abstractmethod

class ToolBase(ABC):
    """
    Base class for tools
    """
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, query: str) -> str:
        pass
