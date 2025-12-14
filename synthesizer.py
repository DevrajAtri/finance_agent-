import json
import os
from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from state import AgentState 

_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

SYSTEM_PROMPT = """
You are the "Synthesizer" for a professional financial analysis agent.
Your goal is to write a professional, data-driven answer based *only* on the provided context.

**INPUT STRUCTURE:**
You will see three sections in your prompt:
1. **SPECIAL INSTRUCTION:** (Optional) High-priority guidance from the Router (e.g., "Refuse this request").
2. **TOOL RESULTS:** Hard facts, math calculations, and live data. **Trust these numbers 100%.**
3. **RETRIEVED DOCUMENTS:** Text excerpts from 10-Ks or transcripts.

**YOUR INSTRUCTIONS:**

1.  **Prioritize Math over Text:**
    * If a Tool Result says "Revenue = $50M" but a Text Document says "Revenue approx $45M", use the **Tool Result**. It is the verified calculation.

2.  **Cite Your Sources:**
    * When using information from the "RETRIEVED DOCUMENTS" section, you MUST cite the source using the header provided in the text.
    * *Format:* `(Source: [File Name], Page [Page Number])`
    * *Example:* "The company faces currency risks (Source: FTNT_10k_2023.pdf, Page 12)."

3.  **Handle "Conversation" Mode:**
    * If the "SPECIAL INSTRUCTION" says to handle a greeting (e.g., "hello"), respond naturally and briefly. Do not look for financial data that isn't there.

4.  **Handle Errors:**
    * If a Tool Result has `STATUS: FAILED`, explain the error politely to the user. Do not try to guess the number.

5.  **Tone & Style:**
    * Be concise, objective, and professional.
    * Use bullet points for lists.
    * If the context is empty or insufficient, admit it. Say: "I don't have enough information to answer that."
"""

def _clean_tool_output(output: Dict[str, Any]) -> str:
    """
    Converts raw tool JSON into a clean, human-readable narrative.
    """
    # 1. Identify the tool and input
    tool_name = output.get("tool_name", "Unknown Tool")
    inputs = output.get("input", "N/A")
    result = output.get("result", output) # Fallback if structure is flat
    status = output.get("status", "unknown")
    
    # 2. Format the Output
    formatted_out = f"   * Action: Ran '{tool_name}' with input: {inputs}\n"
    
    if status == "error" or "error" in str(result).lower():
        formatted_out += f"   * STATUS: FAILED\n"
        formatted_out += f"   * Error Details: {result}\n"
    else:
        # If result is simple, show it. If complex, the LLM parses the string rep.
        formatted_out += f"   * STATUS: SUCCESS\n"
        formatted_out += f"   * Result: {result}\n"
        
    return formatted_out

def _clean_chunk(chunk: Dict[str, Any], index: int) -> str:
    """
    Extracts rich metadata to create a trustworthy Source Header.
    """
    meta = chunk.get("metadata", {})
    source = chunk.get("source", "Unknown")
    text = chunk.get("text", "").strip()
    
    # 1. Extract Trust Signals (File Name, Page)
    filename = meta.get("file_name") or meta.get("filename") or source
    page = meta.get("page_label") or meta.get("page_number") or "N/A"
    
    # 2. Build the Citation Header
    header = f"[Source ID: {index} | File: {filename} | Page: {page}]"
    
    return f"{header}\n{text}\n"

def _format_state_for_synthesis(state: AgentState) -> str:
    """
    Prepares the 'Messy State' for the LLM by cleaning, ordering, and validating data.
    """
    
    # 1. Router Instructions 
    instruction_block = ""
    if instruction := state.get("ins_for_synthesizer"): 
        instruction_block = (
            "!!! SPECIAL INSTRUCTION FROM ROUTER !!!\n"
            f"{instruction}\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n"
        )

    # 2. User Query (Context)
    query_block = f"Original User Query: {state['query']}\n\n"
    
    # 3. Format Tool Outputs
    tool_outputs = state.get("tool_outputs", [])
    if tool_outputs:
        tools_block = "--- TOOL & CALCULATION RESULTS (Trust these numbers) ---\n"
        for i, out in enumerate(tool_outputs, 1):
            tools_block += f"[Result {i}]\n"
            tools_block += _clean_tool_output(out)
            tools_block += "--------------------------------------------------------\n"
    else:
        tools_block = "--- TOOL RESULTS ---\n(No calculations were performed.)\n\n"

    # 4. Format Retrieved Chunks 
    chunks = state.get("retrieved_chunks", [])
    if chunks:
        chunks_block = "--- RETRIEVED TEXT DOCUMENTS (Cite these sources) ---\n"
        for i, chunk in enumerate(chunks, 1):
            chunks_block += _clean_chunk(chunk, i)
            chunks_block += "-----------------------------------------------------\n"
    else:
        chunks_block = "--- RETRIEVED TEXT ---\n(No documents were found.)\n\n"

    # 5. The "Void" Check (Anti-Hallucination Safety)
    # If both major data sources are empty, we inject a warning.
    void_warning = ""
    if not tool_outputs and not chunks and not instruction:
        void_warning = (
            "\n*** SYSTEM ALERT: NO DATA AVAILABLE ***\n"
            "Both tool outputs and retrieved documents are empty.\n"
            "You MUST politely tell the user you could not find the information.\n"
            "Do NOT hallucinate an answer.\n"
        )

    # 6. Assemble Final Prompt
    return (
        f"{instruction_block}"
        f"{query_block}"
        f"{void_warning}"
        f"{tools_block}\n"
        f"{chunks_block}"
    )

def synthesizer(state: AgentState) -> Dict[str, Any]:
    """
    This is the final "synthesizer" node.
    It takes the complete agent state, formats it, and asks the LLM
    to generate the final answer.
    """
    
    # --- 1. SHORT-CIRCUIT CHECK (The "Exit" Mode) ---
    # If the Router explicitly said "EXIT", we skip the LLM entirely.
    instruction = state.get("ins_for_synthesizer", "")
    
    if instruction == "EXIT":
        print("--- SYNTHESIZER: Short-circuiting (Exit Mode) ---")
        final_msg = "I apologize, but that request is out of my scope. I can only analyze the 5 covered cybersecurity companies."
        print(f"\n==================================================")
        print(f" FINAL ANSWER (Short Circuit):\n{final_msg}")
        print(f"==================================================\n")
        return {
            "final_answer": final_msg,
            "retrieved_chunks": [] # Wipe memory to be safe
        }

    # --- 2. PREPARE DATA ---
    formatted_prompt = _format_state_for_synthesis(state)
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=formatted_prompt)
    ]

    print("--- SYNTHESIZER: Generating final answer ---")
    
    try:
        # --- 3. CALL LLM ---
        ai_response = _llm.invoke(messages)
        final_answer = ai_response.content if hasattr(ai_response, 'content') else str(ai_response)

        print(f"--- SYNTHESIZER: Final Answer Generated ---")
        print(f"\n==================================================")
        print(f" FINAL ANSWER:\n{final_answer}")
        
        return {
            "final_answer": final_answer,
            "retrieved_chunks": [],    # Clear memory
            "tool_outputs": [],        # Clear old tool results
            "loop_step": 6,            # Reset the circuit breaker
            "retrieval_count": 0,      # Reset the retrieval limit
            "ins_for_synthesizer": ""  # Clear instructions
        }

    except Exception as e:
        print(f"\n!!! SYNTHESIZER ERROR !!!")
        print(f"Error calling LLM: {e}")
        
        error_answer = (
            "I'm sorry, I encountered an internal error while trying to "
            f"generate the final answer. The error was: {str(e)}"
        )
        return {
            "final_answer": error_answer,
            "retrieved_chunks": []
        }