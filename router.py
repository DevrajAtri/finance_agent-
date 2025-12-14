import os
from typing import Literal, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# Import state definition
from state import AgentState 

LIMITS = {
    "max_retrievals": 3,
    "max_loops": 6
}

_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

class RouteChoice(BaseModel):
    next_step: Literal["retrieve", "plan_tools", "synthesize", "conversational"] = Field(
        description="The next node to route to. Use 'conversational' ONLY for non-financial refusals or greetings."
    )
    explanation: str = Field(description="Brief reason for the decision.")
    
    search_filters: Optional[Dict[str, Any]] = Field(
        description="Only for 'retrieve'. Example: {'ticker': 'FTNT'}. Leave empty for general concepts.",
        default=None
    )
    
    search_namespaces: List[Literal["filings", "transcripts", "textbook", "glossary"]] = Field(
        description="Select the data sources to search. Use ['textbook', 'glossary'] for definitions. Use ['filings', 'transcripts'] for company data.",
        default=["filings", "transcripts", "textbook", "glossary"]
    )

BASE_SYSTEM_PROMPT = """
You are a specialized Financial Analysis Agent.
Your goal is to answer financial queries using your tools and knowledge base.

**YOUR KNOWLEDGE BASE:**
1. **Covered Companies (Deep Data):** Fortinet (FTNT), CrowdStrike (CRWD), Palo Alto Networks (PANW), SentinelOne (S), Zscaler (ZS).
2. **General Finance:** You have textbooks and glossaries for concepts (WACC, EBITDA, etc.).

**ROUTING RULES:**

1. `retrieve`: **Context & Fundamentals.**
   - **Trigger:** "Compare Earnings", "What is WACC?", "Business Strategy".
   - **Smart Namespaces:** - If asking for definitions ("What is ROE?"), select `['textbook', 'glossary']`.
     - If asking for company data ("FTNT Revenue"), select `['filings', 'transcripts']`.
   - **CRITICAL:** Look at the "RETRIEVED CONTENT PREVIEW" below. If you see the answer there, **DO NOT** retrieve again. Route to `synthesize`.

2. `plan_tools`: **Live Market Data & Math.**
   - **Trigger:** "Stock price", "Calculate WACC", "Math".
   - **Retry Policy:** If a tool failed (see Tool Logs), check the error message and try `plan_tools` again with corrected inputs (e.g., fixing "Soon" to "500000").

3. `synthesize`: **Answer Generation.**
   - **Trigger:** You have sufficient information to answer.
   - **Trigger:** You have searched multiple times and cannot find more. It is better to answer with what you have.

4. `conversational`: **Greetings & Absurd Refusals.**
   - **Trigger:** Social greetings ("Hi") or completely non-financial requests ("Bake a cake").

**PRIORITIES:**
- **Answer Confidence:** If you have the data in the snippets, stop searching.
- **Scope:** Do not calculate WACC in your head; use `plan_tools`.
"""

def _format_state_for_llm(state: AgentState) -> str:
    # 1. Semantic History
    history = state.get("chat_history", [])
    recent_history = history[-3:] if len(history) > 3 else history
    history_str = "\n".join(f"  {role}: {text}" for role, text in recent_history)
    
    # 2. Context Check (THE FIX: Show Snippets)
    chunks = state.get("retrieved_chunks", [])
    chunks_preview = ""
    if chunks:
        chunks_preview = "--- RETRIEVED CONTENT PREVIEW (First 5) ---\n"
        for i, c in enumerate(chunks[:5]): 
            # Slice first 150 chars to give the Router a "peek"
            content = c.page_content if hasattr(c, 'page_content') else str(c)
            snippet = content[:150].replace("\n", " ") 
            chunks_preview += f"[{i+1}] ...{snippet}...\n"
        if len(chunks) > 5:
            chunks_preview += f"... (+{len(chunks)-5} more items)\n"
    else:
        chunks_preview = "Retrieved Content: None (Empty).\n"
    
    # 3. Tool Logs
    tool_outputs = state.get("tool_outputs", []) or []
    tools_log = ""
    if tool_outputs:
        tools_log = "--- TOOL EXECUTION LOG ---\n"
        for i, output in enumerate(tool_outputs, 1):
            t_name = output.get("tool_name", "Unknown")
            t_result = output.get("result", output)
            error = output.get("error", None)
            
            log_entry = f"Step {i}: Tool '{t_name}'"
            if error:
                log_entry += f" -> FAILED: {error}"
            else:
                res_str = str(t_result)
                if len(res_str) > 300: res_str = res_str[:300] + "..."
                log_entry += f" -> Success: {res_str}"
            
            tools_log += log_entry + "\n"
    else:
        tools_log = "Tool Outputs: None.\n"

    return (
        f"Query: {state['query']}\n"
        f"Recent Chat:\n{history_str}\n"
        f"\n--- WORKBENCH STATE ---\n"
        f"{chunks_preview}\n" 
        f"{tools_log}"
    )

def router(state: AgentState) -> Dict[str, Any]:
    # Terminal Visibility
    print(f"\n==================================================")
    print(f" USER QUERY: {state['query']}")
    print(f"==================================================")
    print("--- ROUTER: Assessing State ---")
    
    # 1. Global Circuit Breaker (Max Loops)
    current_step = state.get("loop_step", LIMITS["max_loops"])
    new_step_count = current_step - 1
    
    if new_step_count <= 0:
        print("!!! CIRCUIT BREAKER: Max loops reached. Proceeding to Answer. !!!")
        return {
            "next_step": "synthesize", 
            "loop_step": 0
        }

    # 2. Call LLM
    structured_llm = _llm.with_structured_output(RouteChoice)
    messages = [
        SystemMessage(content=BASE_SYSTEM_PROMPT),
        HumanMessage(content=_format_state_for_llm(state))
    ]

    try:
        response = structured_llm.invoke(messages)
        decision = response.next_step
        
        # 3. THE SOFT LIMIT (Block Search Only)
        retrieval_count = state.get("retrieval_count", 0)
        
        if decision == "retrieve" and retrieval_count >= LIMITS["max_retrievals"]:
            print(f"--- ROUTER: Retrieval Limit Hit ({retrieval_count}). Blocking Search. ---")
            decision = "synthesize"

        # 4. EXIT STRATEGY
        if decision == "conversational":
            print(f"--- ROUTER: Conversational/Exit Triggered ---")
            final_msg = response.explanation
            if "out of scope" in final_msg.lower():
                final_msg = "I apologize, but I am a specialized financial agent. I can only assist with market data, financial concepts, and analysis of the 5 covered cybersecurity companies."
                print(f"\n{final_msg}\n")
            return {"final_answer": final_msg, "next_step": "end"}

        print(f"--- ROUTER DECISION: {decision.upper()} ---")
        
        # 5. Update State
        patch = {
            "next_step": decision,
            "loop_step": new_step_count,
        }
        
        if decision == "retrieve":
            patch["retrieval_count"] = retrieval_count + 1
            if response.search_filters:
                patch["retrieval_filters"] = response.search_filters
            
            # Apply the Smart Namespaces selected by the LLM
            patch["search_namespaces"] = response.search_namespaces
            
            print(f"    Target: {response.search_namespaces} | Filters: {response.search_filters}")

        return patch

    except Exception as e:
        print(f"Router Error: {e}")
        return {"next_step": "synthesize", "loop_step": new_step_count}