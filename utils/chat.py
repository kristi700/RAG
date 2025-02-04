from typing import List, Dict
from utils.llm_wrapper import LLM_wrapper
from graph_db.graph_db import NebulaHandler
from rag.context_retrieval import get_context
from vector_db.vector_db import WeaviateVectorDatabase

def build_chat_prompt(conversation_history: List[str], current_query: str, context: Dict[str, str]):
    """
    Build a composite prompt for the LLM.
    """
    system_instruction = "You are an assistant that helps the user understand the content of the provided PDF. Use the context when answering."
    history_text = ""
    for turn in conversation_history:
        history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
    
    prompt = (
        f"{system_instruction}\n\n"
        f"Retrieved Context:\n{context['graph_context']}; {context['vector_context']}\n\n"
        f"Conversation History:\n{history_text}\n"
        f"User: {current_query}\nAssistant:"
    )
    return prompt

# NOTE - maybe context should be updated!
def chat_loop(llm: LLM_wrapper, graph_db: NebulaHandler, vector_db: WeaviateVectorDatabase):
    """
    Run an interactive multi-turn chat session.
    """
    conversation_history = []
    print("Starting chat session. Type 'exit' to quit.")
    
    while True:
        current_query = input("User: ").strip()
        if current_query.lower() in ("exit", "quit"):
            break
        
        context = get_context(graph_db, vector_db, current_query)
        prompt = build_chat_prompt(conversation_history, current_query, context)
        assistant_response = llm.generate_chat(user_prompt=prompt, context=context)
        
        print("Assistant:", assistant_response)

        conversation_history.append({
            "user": current_query,
            "assistant": assistant_response
        })