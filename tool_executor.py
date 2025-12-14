import json
import pydantic
from typing import Dict, Any, List, Callable, Type, Tuple
from state import AgentState 


from agent_tools.tool_calculator import calculator, CalculatorInput
from agent_tools.tool_GRC import growth_rate_calculator, GrowthRateInput
from agent_tools.tool_ratio import ratio_calculator, RatioInput 
from agent_tools.tool_valuation import valuation_tool, AnyValuationInput
from agent_tools.tool_pandas import pandas_tool, PandasToolInput
from agent_tools.table_qa_tool import table_qa_tool, TableQAInput
from agent_tools.data_fetching_tool import data_fetching_tool, AnyDataFetchingInput

TOOL_MAP = {
    "calculator": (calculator, CalculatorInput),
    "growth_rate_calculator": (growth_rate_calculator, GrowthRateInput),
    "ratio_calculator": (ratio_calculator, RatioInput), 
    "valuation_tool": (valuation_tool, AnyValuationInput),
    "pandas_tool": (pandas_tool, PandasToolInput),
    "table_qa_tool": (table_qa_tool, TableQAInput),
    "data_fetching_tool": (data_fetching_tool, AnyDataFetchingInput),
}

def tool_executor(state: AgentState) -> Dict[str, List[Dict[str, Any]]]:
    """
    Executes the tool calls planned by the 'tool_planner'.
    """
    
    tool_calls = state.get("tool_calls", [])
    
    if not tool_calls:
        return {"tool_outputs": []}
        
    print(f"--- TOOL EXECUTOR: Running {len(tool_calls)} tool(s) ---")
    
    tool_outputs = []
    
    for tool_call in tool_calls:
        
        print(f"    Raw Tool Call: {tool_call}")
        
        tool_name = tool_call.get("tool_name") or tool_call.get("name")
        
        tool_args = tool_call.get("args") or tool_call.get("arguments") or {}
        
        if not tool_name:
            print("!!! Error: Missing 'tool_name' in plan !!!")
            continue
            
        if tool_name not in TOOL_MAP:
            print(f"!!! Error: Unknown tool '{tool_name}' !!!")
            tool_outputs.append({"error": f"Unknown tool: {tool_name}", "tool_name": tool_name})
            continue
        
        func_to_call, input_schema = TOOL_MAP[tool_name]
        
        try:
            cleaned_args = tool_args
            if len(tool_args) == 1:
                key = list(tool_args.keys())[0]
                val = tool_args[key]
                if key in ["tool_input", "input", "arguments"] and isinstance(val, dict):
                    print(f"    [Fix] Unwrapping '{key}' layer...")
                    cleaned_args = val
            
            print(f"    Calling {tool_name} with: {cleaned_args}")

            if not cleaned_args:
                print(f"!!! Error: Planner produced empty arguments for {tool_name} !!!")
                tool_outputs.append({
                    "error": "Planner produced empty arguments for tool execution.",
                    "tool_name": tool_name,
                    "status": "invalid_tool_call"
                })
                continue
            validated_input = input_schema.model_validate(cleaned_args)
            raw_result = func_to_call(validated_input)
            
            output = {
                "tool_name": tool_name,
                "input": cleaned_args,
                "result": raw_result,
                "status": "success"
            }
            
        except pydantic.ValidationError as e:
            print(f"!!! Validation Error for {tool_name}: {e} !!!")
            output = {
                "error": f"Validation Error: {str(e)}",
                "tool_name": tool_name,
                "input": tool_args,
                "status": "validation_error"
            }
        except Exception as e:
            print(f"!!! Runtime Error in {tool_name}: {e} !!!")
            output = {
                "error": f"Runtime Error: {str(e)}",
                "tool_name": tool_name,
                "input": tool_args,
                "status": "runtime_error"
            }
            
        tool_outputs.append(output)

    print(f"--- TOOL EXECUTOR: Finished all tools ---")
    return {"tool_outputs": tool_outputs}