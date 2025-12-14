from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any


class GrowthRateInput(BaseModel):
    """
    Schema for the growth_rate_calculator tool's inputs.
    """
    start_value: float = Field(
        description="The initial value (denominator). E.g., last year's revenue."
    )
    end_value: float = Field(
        description="The final value (numerator). E.g., this year's revenue."
    )

    @field_validator('start_value')
    @classmethod  
    def check_start_value_not_zero(cls, v: float) -> float:
        """
        Validates that 'start_value' is not zero to prevent division errors.
        """
        if v == 0.0:
            raise ValueError("start_value cannot be zero (division by zero).")
        return v


def growth_rate_calculator(tool_input: GrowthRateInput) -> Dict[str, Any]:
    """
    Compute the percentage change between two values (end vs. start).
    
    This tool follows the I/O contract:
    - Input: A Pydantic model ('GrowthRateInput') with validated arguments.
    - Output: A dictionary with either a 'result' or 'error' key.
    """
    try:
        
        start_value = tool_input.start_value
        end_value = tool_input.end_value
        change = end_value - start_value
        pchange = change / start_value
        result_rate = round(pchange, 4) 

        return {
            "result": result_rate,
            "growth_rate_decimal": result_rate,
            "growth_rate_percent": f"{result_rate * 100:.2f}%",
            "start_value": start_value,
            "end_value": end_value
        }

    except (ValueError, TypeError) as e:
        
        return {"error": str(e)}
    except Exception as e:
        
        return {"error": f"An unexpected error occurred: {str(e)}"}