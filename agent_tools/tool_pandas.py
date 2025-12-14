import pandas as pd
import pydantic
from typing import List, Dict, Any, Literal, Union, Optional
from pydantic import BaseModel, Field, model_validator
import io
import os
import hashlib
import re


DATA_ROOT = os.path.abspath(os.getenv("PANDAS_DATA_ROOT", "./csv_data"))


os.makedirs(DATA_ROOT, exist_ok=True)


class SelectOp(BaseModel):
    action: Literal["select"]
    columns: List[str] = Field(description="List of column names to keep.")

class FilterCondition(BaseModel):
    col: str = Field(description="Column name to filter on.")
    op: Literal["==", "!=", ">", "<", ">=", "<=", "isin", "notin", "contains"] = Field(
        description="The comparison operator."
    )
    value: Any = Field(description="The value to compare against. For 'isin'/'notin', this should be a list.")

class FilterOp(BaseModel):
    action: Literal["filter"]
    conditions: List[FilterCondition] = Field(description="One or more conditions to filter the DataFrame.")

class SortOp(BaseModel):
    action: Literal["sort"]
    by: List[str] = Field(description="List of column names to sort by.")
    ascending: Union[bool, List[bool]] = Field(True, description="Sort order. True for ascending, False for descending.")

class HeadOp(BaseModel):
    action: Literal["head"]
    n: int = Field(5, description="Number of rows to return from the top.")

class TailOp(BaseModel):
    action: Literal["tail"]
    n: int = Field(5, description="Number of rows to return from the bottom.")

class GroupByAggOp(BaseModel):
    action: Literal["groupby_agg"]
    by: List[str] = Field(description="List of columns to group by.")
    aggregations: Dict[str, Union[str, List[str]]] = Field(
        description="Dictionary mapping column names to aggregation functions (e.g., 'sum', 'mean', 'count')."
    )

class RenameOp(BaseModel):
    action: Literal["rename"]
    columns: Dict[str, str] = Field(description="Dictionary mapping {'old_name': 'new_name'}.")

class DropNAOp(BaseModel):
    action: Literal["dropna"]
    subset: Optional[List[str]] = Field(None, description="Columns to consider when dropping rows with NA values.")

class FillNAOp(BaseModel):
    action: Literal["fillna"]
    value: Any = Field(description="The value to use for filling NA.")
    subset: Optional[List[str]] = Field(None, description="Columns to fill NA values in.")

class ComputeOp(BaseModel):
    action: Literal["compute"]
    new_col: str = Field(description="Name of the new column to create.")
    col1: str = Field(description="The first column name for the operation.")
    op: Literal["+", "-", "*", "/"] = Field(description="The arithmetic operation.")
    col2: Union[str, int, float] = Field(description="The second column name or a scalar value.")

class PivotOp(BaseModel):
    action: Literal["pivot"]
    index: str = Field(description="Column to use as new DataFrame's index.")
    columns: str = Field(description="Column to use to make new DataFrame's columns.")
    values: str = Field(description="Column(s) to use for populating new DataFrame's values.")

class MeltOp(BaseModel):
    action: Literal["melt"]
    id_vars: List[str] = Field(description="Column(s) to use as identifier variables.")
    value_vars: List[str] = Field(description="Column(s) to unpivot.")
    var_name: str = Field("variable", description="Name to use for the 'variable' column.")
    value_name: str = Field("value", description="Name to use for the 'value' column.")


AnyOperation = Union[
    SelectOp, FilterOp, SortOp, HeadOp, TailOp, GroupByAggOp,
    RenameOp, DropNAOp, FillNAOp, ComputeOp, PivotOp, MeltOp
]


class PandasToolInput(BaseModel):
    """
    Input schema for the pandas_tool.
    The tool can load a CSV from either a sandboxed 'data_path'
    or from raw 'csv_text', but not both.
    """
    data_path: Optional[str] = Field(None, description="Relative path to a CSV file in the sandboxed data root.")
    csv_text: Optional[str] = Field(None, description="A string containing the raw CSV data.")
    operations: List[pydantic.SerializeAsAny[AnyOperation]] = Field(
        description="A list of analysis operations to perform in sequence."
    )
    max_output_rows: int = Field(100, description="The maximum number of rows to return in the final CSV.")

    @model_validator(mode='after')
    def check_data_source(self) -> 'PandasToolInput':
        """Validates that exactly one data source is provided."""
        if self.data_path and self.csv_text:
            raise ValueError("Provide either 'data_path' or 'csv_text', not both.")
        if not self.data_path and not self.csv_text:
            raise ValueError("Must provide one of 'data_path' or 'csv_text'.")
        return self
# --- 2. HELPER FUNCTIONS ---

def _read_csv(data_root: str, data_path: Optional[str], csv_text: Optional[str]) -> (pd.DataFrame, Dict[str, Any]):
    """Loads a DataFrame and generates initial metadata."""
    meta = {}
    df = pd.DataFrame()
    if data_path:
        
        full_path = os.path.abspath(os.path.join(data_root, data_path))
        if not full_path.startswith(data_root):
            raise PermissionError(f"Access denied: Path '{data_path}' is outside the allowed data directory.")
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found at '{data_path}'.")
        
        df = pd.read_csv(full_path)
        meta["source_type"] = "path"
        meta["source_name"] = data_path
        with open(full_path, 'rb') as f:
            meta["file_hash_sha1"] = hashlib.sha1(f.read()).hexdigest()
            
    elif csv_text:
        df = pd.read_csv(io.StringIO(csv_text))
        meta["source_type"] = "text"
        meta["source_name"] = "csv_text"
        meta["file_hash_sha1"] = hashlib.sha1(csv_text.encode('utf-8')).hexdigest()
        
    meta["initial_rows"] = len(df)
    meta["initial_cols"] = len(df.columns)
    return df, meta

def _normalize_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Converts object columns that look numeric into actual numbers."""
    for col in df.select_dtypes(include=['object']).columns:
        try:
            cleaned_col = (
                df[col]
                .astype(str)
                .str.replace(r"[\$,]", "", regex=True)
                .str.strip()
                .str.replace(r"\((.*)\)", r"-\1", regex=True)
            )
            
            numeric_col = pd.to_numeric(cleaned_col, errors='coerce')
            
            if not numeric_col.isnull().all() and numeric_col.count() > 0:
                df[col] = numeric_col
        except Exception:
            
            pass
    return df

def _build_colmap(df: pd.DataFrame) -> Dict[str, str]:
    """Creates a case-insensitive lookup map for column names."""
   
    return {col.lower(): col for col in reversed(df.columns)}

def _resolve_col(name: str, colmap: Dict[str, str]) -> str:
    """Finds the real column name from a case-insensitive name."""
    real_name = colmap.get(name.lower())
    if not real_name:
        raise ValueError(f"Column '{name}' not found. Available columns: {list(colmap.values())}")
    return real_name


def _do_select(df: pd.DataFrame, colmap: Dict[str, str], op: SelectOp) -> (pd.DataFrame, Dict[str, Any]):
    resolved_cols = [_resolve_col(c, colmap) for c in op.columns]
    return df[resolved_cols], {}

def _do_filter(df: pd.DataFrame, colmap: Dict[str, str], op: FilterOp) -> (pd.DataFrame, Dict[str, Any]):
    mask = pd.Series(True, index=df.index)
    for c in op.conditions:
        col = _resolve_col(c.col, colmap)
        op_str = c.op
        val = c.value
        
        if op_str == "==": mask &= (df[col] == val)
        elif op_str == "!=": mask &= (df[col] != val)
        elif op_str == ">": mask &= (df[col] > val)
        elif op_str == "<": mask &= (df[col] < val)
        elif op_str == ">=": mask &= (df[col] >= val)
        elif op_str == "<=": mask &= (df[col] <= val)
        elif op_str == "isin": mask &= (df[col].isin(val))
        elif op_str == "notin": mask &= (~df[col].isin(val))
        elif op_str == "contains": mask &= (df[col].astype(str).str.contains(str(val), case=False, na=False))
        else: raise ValueError(f"Unsupported filter operation: {op_str}")
            
    return df[mask], {"filtered_rows": len(df[mask])}

def _do_sort(df: pd.DataFrame, colmap: Dict[str, str], op: SortOp) -> (pd.DataFrame, Dict[str, Any]):
    by_cols = [_resolve_col(c, colmap) for c in op.by]
    return df.sort_values(by=by_cols, ascending=op.ascending), {}

def _do_head(df: pd.DataFrame, colmap: Dict[str, str], op: HeadOp) -> (pd.DataFrame, Dict[str, Any]):
    return df.head(op.n), {}

def _do_tail(df: pd.DataFrame, colmap: Dict[str, str], op: TailOp) -> (pd.DataFrame, Dict[str, Any]):
    return df.tail(op.n), {}

def _do_groupby_agg(df: pd.DataFrame, colmap: Dict[str, str], op: GroupByAggOp) -> (pd.DataFrame, Dict[str, Any]):
    by_cols = [_resolve_col(c, colmap) for c in op.by]
    
    
    resolved_aggs = {}
    for col, agg_func in op.aggregations.items():
        resolved_col = _resolve_col(col, colmap)
        resolved_aggs[resolved_col] = agg_func
        
    grouped = df.groupby(by=by_cols).agg(resolved_aggs)
    
    
    if isinstance(grouped.columns, pd.MultiIndex):
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        
    return grouped.reset_index(), {}

def _do_rename(df: pd.DataFrame, colmap: Dict[str, str], op: RenameOp) -> (pd.DataFrame, Dict[str, Any]):
    resolved_map = {_resolve_col(old, colmap): new for old, new in op.columns.items()}
    return df.rename(columns=resolved_map), {}

def _do_dropna(df: pd.DataFrame, colmap: Dict[str, str], op: DropNAOp) -> (pd.DataFrame, Dict[str, Any]):
    subset = [_resolve_col(c, colmap) for c in op.subset] if op.subset else None
    return df.dropna(subset=subset), {}

def _do_fillna(df: pd.DataFrame, colmap: Dict[str, str], op: FillNAOp) -> (pd.DataFrame, Dict[str, Any]):
    subset = [_resolve_col(c, colmap) for c in op.subset] if op.subset else None
    if subset:
        df[subset] = df[subset].fillna(value=op.value)
        return df, {}
    else:
        return df.fillna(value=op.value), {}

def _do_compute(df: pd.DataFrame, colmap: Dict[str, str], op: ComputeOp) -> (pd.DataFrame, Dict[str, Any]):
    col1 = df[_resolve_col(op.col1, colmap)]
    col2 = df[_resolve_col(op.col2, colmap)] if isinstance(op.col2, str) else op.col2
    
    if op.op == "+": df[op.new_col] = col1 + col2
    elif op.op == "-": df[op.new_col] = col1 - col2
    elif op.op == "*": df[op.new_col] = col1 * col2
    elif op.op == "/": df[op.new_col] = col1 / col2
    else: raise ValueError(f"Unsupported compute operation: {op.op}")
        
    return df, {}

def _do_pivot(df: pd.DataFrame, colmap: Dict[str, str], op: PivotOp) -> (pd.DataFrame, Dict[str, Any]):
    index = _resolve_col(op.index, colmap)
    columns = _resolve_col(op.columns, colmap)
    values = _resolve_col(op.values, colmap)
    return df.pivot(index=index, columns=columns, values=values).reset_index(), {}

def _do_melt(df: pd.DataFrame, colmap: Dict[str, str], op: MeltOp) -> (pd.DataFrame, Dict[str, Any]):
    id_vars = [_resolve_col(c, colmap) for c in op.id_vars]
    value_vars = [_resolve_col(c, colmap) for c in op.value_vars]
    return df.melt(id_vars=id_vars, value_vars=value_vars, var_name=op.var_name, value_name=op.value_name), {}

_ACTIONS = {
    "select": _do_select,
    "filter": _do_filter,
    "sort": _do_sort,
    "head": _do_head,
    "tail": _do_tail,
    "groupby_agg": _do_groupby_agg,
    "rename": _do_rename,
    "dropna": _do_dropna,
    "fillna": _do_fillna,
    "compute": _do_compute,
    "pivot": _do_pivot,
    "melt": _do_melt,
}

def pandas_tool(tool_input: PandasToolInput) -> Dict[str, Any]:
    """
    A sandboxed pandas DataFrame processor for CSV data.
    
    It loads a CSV from a secure path or text, then executes a
    list of data manipulation operations in sequence.
    
    Returns a dictionary with the resulting CSV string and metadata,
    or a scalar result if an aggregation produces one.
    """
    try:
       
        df, meta_in = _read_csv(DATA_ROOT, tool_input.data_path, tool_input.csv_text)
        
        df = _normalize_numeric_columns(df)
        
        colmap = _build_colmap(df)
        
        operation_log = []
        result_data = df
        
        for op in tool_input.operations:
            handler = _ACTIONS.get(op.action)
            if not handler:
                raise ValueError(f"Unknown operation action: {op.action}")
             
            result_data, op_meta = handler(result_data, colmap, op)
            
            operation_log.append({
                "action": op.action,
                "params": op.model_dump(),
                "meta": op_meta
            })
            
            if not isinstance(result_data, pd.DataFrame):
               
                meta_final = {
                    "meta_in": meta_in, 
                    "operations": operation_log
                }
                return {"result": result_data, "meta": meta_final}
            
            colmap = _build_colmap(result_data)

        df_result = result_data
        total_rows = len(df_result)
        
        if total_rows > tool_input.max_output_rows:
            df_result = df_result.head(tool_input.max_output_rows)
            
        result_csv = df_result.to_csv(index=False)
        preview_head = df_result.head(5).to_csv(index=False)
        
        meta_out = {
            "final_rows_total": total_rows,
            "final_rows_returned": len(df_result),
            "final_cols": len(df_result.columns),
            "available_columns": list(df_result.columns),
            "preview_head": preview_head
        }
        
        return {
            "result_csv": result_csv,
            "meta": {
                "meta_in": meta_in,
                "meta_out": meta_out,
                "operations": operation_log
            }
        }

    except Exception as e:
       
        return {"error": str(e)}