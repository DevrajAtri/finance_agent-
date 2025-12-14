# graph.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()
from state import AgentState
from router import router
from tool_planner import tool_planner
from tool_executor import tool_executor
from retriever import retrieve
from synthesizer import synthesizer
from ask import ask_user
from ingest_user_file import ingest_user_file, IngestInput 

if not os.getenv("PINECONE_API_KEY"):
    print("Warning: PINECONE_API_KEY environment variable not set.")
if not os.getenv("GOOGLE_API_KEY"):
    print("Warning: GOOGLE_API_KEY not set.")


print("Building the agent graph...")
workflow = StateGraph(AgentState)

#  Add Nodes 
workflow.add_node("router", router)
workflow.add_node("retrieve", retrieve)
workflow.add_node("tool_planner", tool_planner)
workflow.add_node("tool_executor", tool_executor)
workflow.add_node("synthesizer", synthesizer)
workflow.add_node("ask_user", ask_user)
workflow.add_node("ingest_user_file", ingest_user_file)

#  Wiring Edges 

def entry_gate(state: AgentState):
    """
    Determines the entry point.
    If a user file is present, we start by ingesting it.
    Otherwise, we go straight to the router.
    """
    if state.get("user_file_info"):
        return "ingest_user_file"
    return "router"

workflow.set_conditional_entry_point(
    entry_gate,
    {
        "ingest_user_file": "ingest_user_file",
        "router": "router"
    }
)

# Connect Ingest to Router 
workflow.add_edge("ingest_user_file", "router")

def get_next_step(state: AgentState):
    return state["next_step"]


workflow.add_conditional_edges(
    "router",
    get_next_step,
    {
        "retrieve": "retrieve",
        "plan_tools": "tool_planner",
        "synthesize": "synthesizer",
        "ask_user": "ask_user",
        "end": END  
    }
)

#  Normal Edges 
workflow.add_edge("retrieve", "router")
workflow.add_edge("tool_planner", "tool_executor")
workflow.add_edge("tool_executor", "router")
workflow.add_edge("ask_user", "router")

workflow.add_edge("synthesizer", END)

print("Compiling graph...")
app = workflow.compile()
print(" Agent graph compiled successfully!")