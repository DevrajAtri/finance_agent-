import json
import os
from typing import Dict, Any, List, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

from state import AgentState 


_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", 
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

SYSTEM_PROMPT = """
You are the "Clarifier" for a financial analysis agent.
The agent has hit a roadblock and cannot proceed. Your *only* job is to formulate a single, clear question to the user to get the necessary information.

You will be given the agent's "workbench" (the state) which shows the user's query and any retrieved data or tool errors.

RULES:
1.  Analyze the 'Original User Query' to understand the goal.
2.  Analyze the 'Workbench' to see *why* the agent is stuck. (e.g., a tool error, missing data for a tool, or an ambiguous query).
3.  Formulate a *single, direct question* to the user.
4.  Do NOT answer the user's query. Do NOT apologize.
5.  Just ask the one question needed to get un-stuck.

Example 1: The query is "Run a DCF" and the workbench is empty.
Your Response: "To run the DCF, what discount rate (WACC) and terminal growth rate should I assume?"

Example 2: A tool output is '{"error": "start_value cannot be zero"}'.
Your Response: "I could not calculate the growth rate because the 'start_value' was zero. Could you provide a valid starting value?"

Example 3: The query is "What's the P/E?" and the retriever found a price but no EPS.
Your Response: "I found the current stock price, but I could not find the Earnings Per Share (EPS). How would you like to proceed?"
"""


def _format_state_for_clarification(state: AgentState) -> str:
    """
    Converts the AgentState into a clean, human-readable
    string for the Clarifier LLM.
    """
    
    query_str = f"Original User Query (The Goal): {state['query']}\n"
    
    chunks_str = "Retrieved Context: None\n" # Default
    if chunks := state.get("retrieved_chunks"):
        chunks_str = "--- Retrieved Context ---\n"
        for i, chunk in enumerate(chunks, 1):
            chunks_str += f"[Chunk {i}] (Source: {chunk.get('source', 'N/A')})\n"
            chunks_str += f"  Text: {chunk.get('text', '')[:250]}...\n" 
            chunks_str += "-------------------------\n"
            
    tools_str = "Calculation Results: None\n" # Default
    if outputs := state.get("tool_outputs"):
        tools_str = "--- Calculation Results (Check for errors here) ---\n"
        for i, out in enumerate(outputs, 1):
            tools_str += f"[Result {i}]\n"
            tools_str += f"{json.dumps(out, indent=2)}\n"
            tools_str += "-------------------------\n"

    
    return (
        f"{query_str}\n"
        " AGENT'S WORKBENCH (Find the problem here) \n\n"
        f"{chunks_str}\n"
        f"{tools_str}\n"
        "END WORKBENCH "
    )


def ask_user(state: AgentState) -> Dict[str, Any]:
    """
    This is the "ask_user" node.
    It takes the current agent state, formats it, and asks an LLM
    (GenAI) to generate a clarification question.
    
    UPDATES:
    - Now waits for user input via input()
    - Updates chat_history so Router sees the answer
    """
    
    print("--- ASKING USER: Agent is stuck ---")
    
    # 1. Format the state into a readable prompt
    formatted_prompt = _format_state_for_clarification(state)
    
    # 2. Create the prompt messages
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", formatted_prompt)
    ]
    
    try:
        # 3. Call the LLM
        ai_response = _llm.invoke(messages)
        question = ai_response.content if hasattr(ai_response, 'content') else str(ai_response)

        
        # 4. Ask the User and WAIT
        print(f"\n Agent Needs Clarification: {question}")
        print("---------------------------------------")
        
        # This blocks execution until you type in the terminal
        user_answer = input("User (Answer): ")
        
        # 5. Update State
        # We need to append the new Q&A to the history so the Router knows what happened.
        current_history = state.get("chat_history", [])
        
        # Format: (AI Question, Human Answer)
        new_interaction = [
            ("ai", question),
            ("human", user_answer)
        ]
        
        print("--- Input Received. Resuming Workflow... ---")

        return {
            "chat_history": new_interaction, 
            "clarification_question": question
        }

    except Exception as e:
        print(f"\n!!! ASK_USER NODE ERROR !!!")
        print(f"Error calling LLM: {e}")
        
        error_question = (
            "I'm sorry, I encountered an internal error while trying to "
            "ask for clarification. Could you please rephrase your request?"
        )
        return {"clarification_question": error_question}