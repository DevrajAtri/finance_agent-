import os
import json
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from agent_tools.tool_valuation import AnyValuationInput
from agent_tools.tool_ratio import RatioInput
from tool_executor import CalculatorInput, AnyDataFetchingInput
from tool_loader import get_tool_signatures

TOOL_DEFINITIONS = {
    "valuation_tool": AnyValuationInput,
    "ratio_calculator": RatioInput,
    "calculator": CalculatorInput,
    "data_fetching_tool": AnyDataFetchingInput,
}

tool_schema_str = get_tool_signatures(TOOL_DEFINITIONS)

_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

SYSTEM_PROMPT = f"""
You are the Tool Planner.

Your job is to decide which tool to call and to produce VALID arguments
that strictly follow the tool signatures.

AVAILABLE TOOLS:
{tool_schema_str}

CRITICAL RULES:
1. **Extraction:** You MUST extract exact values from the user query.
2. **JSON Format:** You MUST provide the arguments as a **valid JSON string** in the `args_json` field.
3. **No Empty Args:** The JSON string must NOT represent an empty dictionary `{{}}`.
4. **Valuation Tool:** You MUST provide the `operation` key (e.g., 'npv', 'wacc').

EXAMPLE:
User: "Calculate NPV for cash flows -100, 50, 60 at 10% rate."

Output:
{{
  "steps": [
    {{
      "reasoning": "Extracted cash_flows [-100, 50, 60] and discount_rate 0.10",
      "tool_name": "valuation_tool",
      "args_json": "{{\\"operation\\": \\"npv\\", \\"cash_flows\\": [-100.0, 50.0, 60.0], \\"discount_rate\\": 0.10}}"
    }}
  ]
}}
"""

class ToolCall(BaseModel):
    reasoning: str = Field(description="Explain what values you extracted")
    tool_name: str = Field(description="Exact tool name")
    args_json: str = Field(description="The arguments as a valid JSON string. Example: '{\"key\": \"value\"}'")

class ToolPlan(BaseModel):
    steps: List[ToolCall]

def tool_planner(state: Dict[str, Any]) -> Dict[str, Any]:
    print("--- TOOL PLANNER: Analyzing state ---")

    query = state.get("query", "")
    if not query:
        return {"tool_calls": []}

    planner_llm = _llm.with_structured_output(ToolPlan)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Query: {query}"),
    ]

    try:
        print("--- TOOL PLANNER: Invoking LLM ---")
        plan = planner_llm.invoke(messages)
        print("--- TOOL PLANNER: LLM Responded ---")

        if not plan or not plan.steps:
            print("--- TOOL PLANNER: No steps returned ---")
            return {"tool_calls": []}

        valid_steps = []

        for step in plan.steps:
            
            try:
                parsed_args = json.loads(step.args_json)
            except json.JSONDecodeError:
                print(f"!!! Error parsing JSON for {step.tool_name}: {step.args_json}")
                continue 

            print(f"    [Plan] {step.tool_name}")
            print(f"           Reasoning: {step.reasoning}")
            print(f"           Args: {parsed_args}")

            if not parsed_args or not isinstance(parsed_args, dict):
                print("    [Planner Warning] Empty or invalid args. Skipping tool call.")
                continue

            valid_steps.append({
                "tool_name": step.tool_name,
                "args": parsed_args, # We store the PARSED dict for the Executor
            })

        if not valid_steps:
            print("--- TOOL PLANNER: No valid tool calls after validation ---")
            return {"tool_calls": []}

        return {"tool_calls": valid_steps}

    except Exception as e:
        print(f"--- TOOL PLANNER ERROR: {e} ---")
        return {"tool_calls": []}