# tool_loader.py
import typing
from typing import Type, get_args, get_origin, Union, Literal
from pydantic import BaseModel, RootModel


def _format_field_type(field_type) -> str:
    """Helper to convert Python types to readable strings."""
    # Handle Optionals: Optional[float] 
    if get_origin(field_type) is Union:
        args = get_args(field_type)
        if type(None) in args:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return f"{_format_field_type(non_none[0])}?"
    
    if get_origin(field_type) is list or get_origin(field_type) is typing.List:
        args = get_args(field_type)
        if args:
            return f"List[{_format_field_type(args[0])}]"
        return "List"

    if get_origin(field_type) is Literal:
        return str(get_args(field_type))

    if hasattr(field_type, "__name__"):
        return field_type.__name__
    
    return str(field_type)

def _generate_sig_from_model(tool_name: str, model_cls: Type[BaseModel]) -> str:
    """Generates a single line signature from a flat BaseModel."""
    parts = []
    schema = model_cls.model_fields
    
    for name, field in schema.items():
        if name == "operation" and get_origin(field.annotation) is Literal:
            
            val = get_args(field.annotation)[0]
            parts.insert(0, f"operation='{val}'") 
            continue
            
        type_str = _format_field_type(field.annotation)
        is_required = field.is_required()
        req_marker = "*" if is_required else ""
        
        parts.append(f"{name}{req_marker}: {type_str}")
    
    return f"- {tool_name}({', '.join(parts)})"

def get_tool_signatures(tool_map: dict[str, Type[BaseModel]]) -> str:
    """
    Main function to generate the cheat sheet.
    Args:
        tool_map: Dict mapping function name to Input Model
                  e.g., {"valuation_tool": AnyValuationInput}
    """
    signatures = []
    
    for tool_name, model_cls in tool_map.items():
        if issubclass(model_cls, RootModel):
            root_field = model_cls.model_fields["root"]
            annotation = root_field.annotation
            
            if get_origin(annotation) is Union:
                sub_models = get_args(annotation)
                for sub in sub_models:
                    
                    signatures.append(_generate_sig_from_model(tool_name, sub))
            else:
                # Fallback if RootModel isn't a Union
                signatures.append(_generate_sig_from_model(tool_name, model_cls))
                
        # CASE 2: Standard Flat Model (The Ratio Pattern)
        else:
            signatures.append(_generate_sig_from_model(tool_name, model_cls))
            
    return "\n".join(signatures)