import operator
from typing import Annotated, List, Dict, Any, Optional, Tuple, TypedDict

class AgentState(TypedDict):
    """
    The central state object for the financial agent.
    """
    # --- Core flow ---
    query: str
    chat_history: Annotated[List[Tuple[str, str]], operator.add]
    
    # --- Context / Data ---
    ins_for_synthesizer: Optional[str]
    retrieved_chunks: List[Dict[str, Any]]
    
    # --- Search Scope ---
    search_namespaces: List[str]
    retrieval_filters: Optional[Dict[str, Any]] 
    
    # --- Tool Management ---
    tool_calls: List[Dict[str, Any]]   
    tool_outputs: Annotated[List[Dict[str, Any]], operator.add]
    
    # --- User-Uploaded File Management ---
    user_file_info: Optional[Dict[str, Any]] 
    
    # --- Interaction Management ---
    clarification_question: Optional[str] 
    final_answer: Optional[str] 
    
    # --- Routing ---
    next_step: Optional[str]

    # --- Counters & Limits ---
    loop_step: int          
    retrieval_count: int    
    
    # [DEPRECATED but KEPT]: Kept to prevent crashes in older nodes if they try to read them.
    # The Router now relies on 'loop_step' instead.
    tool_error_count: int
    ask_user_count: int
    tool_use_count: int