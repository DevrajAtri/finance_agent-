import os
import pandas as pd
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from transformers import pipeline, Pipeline

# --- CONFIGURATION ---

TABLE_DATA_ROOT = os.path.abspath(
    os.getenv("TABLE_DATA_ROOT", "structured_tables")
)

# Create the directory if it doesn't exist 
os.makedirs(TABLE_DATA_ROOT, exist_ok=True)


_tapas_pipeline: Optional[Pipeline] = None
_tapas_model_name = "google/tapas-base-finetuned-wtq"

def _ensure_tapas_models():
    """
    Lazy-loads the TAPAS pipeline on first use.
    """
    global _tapas_pipeline
    if _tapas_pipeline is None:
        try:
            print(f"Lazy-loading TAPAS model: {_tapas_model_name}...")
            _tapas_pipeline = pipeline(
                "table-question-answering",
                model=_tapas_model_name,
                tokenizer=_tapas_model_name
            )
            print("TAPAS model loaded successfully.")
        except ImportError:
            raise ImportError(
                "TAPAS dependencies not found. Please run: "
                "pip install transformers torch"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load TAPAS model: {e}")

#  1. PYDANTIC SCHEMA 

class TableQAInput(BaseModel):
    """
    Input schema for the table_qa_tool.
    """
    query: str = Field(
        description="The natural language question to ask the table."
    )
    csv_path: str = Field(
        description=(
            "The relative path to the structured .csv file. "
            "This path is found in the 'table_struct_path' [cite: 52] metadata of a 'table_text' chunk."
        )
    )

#  2. THE TOOL IMPLEMENTATION 

def table_qa_tool(tool_input: TableQAInput) -> Dict[str, Any]:
    """
    Answers a natural language query about a specific structured table
    using the TAPAS model.
    
    This tool follows the I/O contract:
    - Input: A Pydantic model ('TableQAInput') with validated arguments.
    - Output: A dictionary with either a 'result' or 'error' key.
    """
    try:
        # 1. Ensure models are loaded
        _ensure_tapas_models()
        if _tapas_pipeline is None:
            
            raise RuntimeError("TAPAS pipeline could not be initialized.")

        # 2. 
        # Verify the CSV path is inside the allowed data root.
        full_path = os.path.abspath(
            os.path.join(TABLE_DATA_ROOT, tool_input.csv_path)
        )
        
        if not full_path.startswith(TABLE_DATA_ROOT):
            raise PermissionError(
                f"Access denied: Path '{tool_input.csv_path}' is outside "
                f"the allowed data directory '{TABLE_DATA_ROOT}'."
            )
            
        if not os.path.exists(full_path):
            raise FileNotFoundError(
                f"File not found: {tool_input.csv_path}. "
                f"Full path checked: {full_path}"
            )

        # 3. Load the table
        
        table_df = pd.read_csv(full_path, dtype=str)

        # 4. Run the TAPAS pipeline
        tapas_result = _tapas_pipeline(
            table=table_df,
            query=tool_input.query
        )
        
        
        #  SUCCESS: Return structured output 
        return {
            "result": tapas_result.get("answer"),
            "source_csv": tool_input.csv_path,
            "full_tapas_output": tapas_result,
            "query": tool_input.query
        }

    except (ValueError, TypeError, PermissionError, FileNotFoundError) as e:
        
        return {
            "error": str(e),
            "inputs": tool_input.model_dump()
        }
    except Exception as e:
        
        return {
            "error": f"An unexpected error occurred in table_qa_tool: {str(e)}",
            "inputs": tool_input.model_dump()
        }