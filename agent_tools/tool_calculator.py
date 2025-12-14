from pydantic import BaseModel, Field, model_validator
from typing import List, Literal, Dict, Any


class CalculatorInput(BaseModel):
    """
    Schema for the calculator tool's inputs.
    """
    operation: Literal["add", "subtract", "multiply", "divide"] = Field(
        description="The arithmetic operation to perform."
    )
    numbers: List[float] = Field(
        description="The numeric values to operate on.",
        min_length=1  
    )

    @model_validator(mode='after')
    def check_number_count_for_ops(self) -> 'CalculatorInput':
        """
        Validates that 'subtract' and 'divide' have at least two numbers.
        """
        
        if self.operation in ("subtract", "divide") and len(self.numbers) < 2:
            raise ValueError(
                f"Operation '{self.operation}' requires at least two numbers."
            )
        return self


def calculator(tool_input: CalculatorInput) -> Dict[str, Any]:
    """
    Perform a basic arithmetic operation on a list of numbers.
    
    This tool follows the I/O contract:
    - Input: A Pydantic model ('CalculatorInput') with validated arguments.
    - Output: A dictionary with either a 'result' or 'error' key.
    """
    try:
        
        operation = tool_input.operation
        numbers = tool_input.numbers
        
        
        if operation == "add":
            result = sum(numbers)

        elif operation == "subtract":
            
            result = numbers[0]
            for n in numbers[1:]:
                result -= n

        elif operation == "multiply":
            result = 1
            for n in numbers:
                result *= n

        elif operation == "divide":
            
            result = numbers[0]
            for n in numbers[1:]:
                if n == 0:
                    
                    raise ValueError("Division by zero is not allowed.")
                result /= n
        return {
            "result": result,
            "operation": operation,
            "numbers": numbers
        }

    except (ValueError, TypeError) as e:
        
        return {"error": str(e)}
    except Exception as e:
        
        return {"error": f"An unexpected error occurred: {str(e)}"}