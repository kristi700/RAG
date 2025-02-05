from typing import List
from utils.llm_wrapper import LLM_wrapper
from graph_db.graph_db import NebulaHandler
from rag.context_retrieval import get_context
from vector_db.vector_db import WeaviateVectorDatabase

def get_history(conversation_history: List[str]):
    history_text = ""
    for turn in conversation_history:
        history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
    return history_text


# NOTE - maybe context should be updated!
def chat_loop(llm: LLM_wrapper, context_agent):
    """
    Run an interactive multi-turn chat session.
    """
    conversation_history = []
    print("Starting chat session. Type 'exit' to quit.")
    
    while True:
        current_query = input("User: ").strip()
        if current_query.lower() in ("exit", "quit"):
            break
        
        context = context_agent.run(current_query)
        history = get_history(conversation_history)
        assistant_response = llm.generate_chat(user_prompt=current_query, context=context, history=history)
        
        print("Assistant:", assistant_response)

        conversation_history.append({
            "user": current_query,
            "assistant": assistant_response
        })