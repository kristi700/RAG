from abc import ABC
from utils.llm_wrapper import LLM_wrapper

class AgentBase(ABC):
    """
    Base class for agents
    """
    def __init__(self, name: str, llm: LLM_wrapper, tools=None, system_prompt: str = None):
        self.llm = llm
        self.name = name
        self.system_prompt = system_prompt or f"You are the {name} agent."
        self.tools = tools if tools is not None else {}
        self.memory = [] # TODO - check for better ways to utilize memory!

    def run(self, user_input: str) -> str:
        prompt = f"{self.system_prompt}\n\nUser: {user_input}\nAgent:"
        response = self.llm.generate_text(prompt)
        return response    


    

    