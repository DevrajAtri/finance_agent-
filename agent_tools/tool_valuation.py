from pydantic import BaseModel, Field, model_validator, RootModel
from typing import List, Dict, Any, Optional, Callable, Union, Literal

def _df(rate: float, t: float, mid_year: bool) -> float:
    shift = 0.5 if mid_year else 0.0
    return (1.0 + rate) ** -(t - shift)

def _npv_from_series(cash_flows: List[float], rate: float, mid_year: bool = False, include_t0: bool = False) -> float:
    if rate <= -1.0: raise ValueError("discount_rate must be > -100%")
    if not cash_flows: return 0.0
    total = 0.0
    if include_t0:
        total += cash_flows[0]
        flows = cash_flows[1:]
        start_t = 1
    else:
        flows = cash_flows
        start_t = 1
    for i, cf in enumerate(flows, start=start_t):
        total += cf * _df(rate, i, mid_year)
    return total

def _irr_bisection(cash_flows: List[float], tol: float = 1e-7, max_iter: int = 200) -> float:
    def npv_at(r: float) -> float:
        if not cash_flows: return 0.0
        total = cash_flows[0]
        for t, cf in enumerate(cash_flows[1:], start=1):
            total += cf / (1.0 + r) ** t
        return total
    low, high = -0.9999, 10.0
    f_low, f_high = npv_at(low), npv_at(high)
    if f_low == 0.0: return low
    if f_high == 0.0: return high
    if f_low * f_high > 0: raise ValueError("IRR not bracketed (cash flows may not have a sign change).")
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        f_mid = npv_at(mid)
        if abs(f_mid) < tol: return mid
        if f_low * f_mid < 0: high, f_high = mid, f_mid
        else: low, f_low = mid, f_mid
    return (low + high) / 2.0

#  Core Operations 
def npv(cash_flows: List[float], discount_rate: float, *, mid_year: bool = False, include_t0: bool = False) -> float:
    return float(_npv_from_series(cash_flows, discount_rate, mid_year, include_t0))

def irr(cash_flows: List[float]) -> float:
    return float(_irr_bisection(cash_flows))

def terminal_value_gordon(fcf_next_year: float, rate: float, g: float) -> float:
    if rate <= g: raise ValueError("rate must be greater than g for Gordon Growth.")
    return float(fcf_next_year / (rate - g))

def terminal_value_exit_multiple(metric_value: float, multiple: float) -> float:
    return float(metric_value * multiple)

def wacc(cost_of_equity: float, cost_of_debt_pre_tax: float, tax_rate: float, equity_value: float, debt_value: float) -> float:
    total = equity_value + debt_value
    if total <= 0: raise ValueError("E + D must be > 0")
    we = equity_value / total
    wd = debt_value / total
    return float(we * cost_of_equity + wd * cost_of_debt_pre_tax * (1.0 - tax_rate))

def cost_of_equity_capm(risk_free: float, beta: float, market_premium: float) -> float:
    return float(risk_free + beta * market_premium)

def dcf_fcff(fcff: List[float], wacc_rate: float, *, terminal: Dict[str, Any], mid_year: bool = True, net_debt: float = 0.0, minority_interest: float = 0.0, cash_and_investments: float = 0.0, shares_outstanding: Optional[float] = None) -> Dict[str, Any]:
    if not fcff: raise ValueError("fcff must contain at least one period.")
    if wacc_rate <= -1.0: raise ValueError("wacc_rate must be > -100%")
    pv_projection = 0.0
    for t, cf in enumerate(fcff, start=1):
        pv_projection += cf * _df(wacc_rate, t, mid_year)
    if terminal.get("method") == "gordon":
        g = float(terminal["g"])
        fcf_next = fcff[-1] * (1.0 + g)
        tv = terminal_value_gordon(fcf_next, wacc_rate, g)
    elif terminal.get("method") == "exit_multiple":
        multiple = float(terminal["multiple"])
        metric = float(terminal["metric"])
        tv = terminal_value_exit_multiple(metric, multiple)
    else: raise ValueError("terminal.method must be 'gordon' or 'exit_multiple'")
    N = len(fcff)
    pv_terminal = tv * _df(wacc_rate, N, mid_year)
    enterprise_value = pv_projection + pv_terminal
    equity_value = enterprise_value - float(net_debt) - float(minority_interest) + float(cash_and_investments)
    out: Dict[str, Any] = {
        "enterprise_value": float(enterprise_value), "equity_value": float(equity_value),
        "pv_projection": float(pv_projection), "terminal_value": float(tv), "pv_terminal": float(pv_terminal),
        "assumptions": {
            "wacc": float(wacc_rate), "mid_year": bool(mid_year), "terminal_method": terminal.get("method"),
            **({"g": float(terminal["g"])} if terminal.get("method") == "gordon" else {}),
            **({"multiple": float(terminal["multiple"]), "metric": float(terminal["metric"])}
               if terminal.get("method") == "exit_multiple" else {}),
        },
        "adjustments": {
            "net_debt": float(net_debt), "minority_interest": float(minority_interest),
            "cash_and_investments": float(cash_and_investments),
        },
    }
    if shares_outstanding and shares_outstanding > 0:
        out["equity_value_per_share"] = float(equity_value / shares_outstanding)
    return out

def dcf_fcfe(fcfe: List[float], cost_of_equity: float, *, terminal: Dict[str, Any], mid_year: bool = True, shares_outstanding: Optional[float] = None) -> Dict[str, Any]:
    if not fcfe: raise ValueError("fcfe must contain at least one period.")
    if cost_of_equity <= -1.0: raise ValueError("cost_of_equity must be > -100%")
    pv_projection = 0.0
    for t, cf in enumerate(fcfe, start=1):
        pv_projection += cf * _df(cost_of_equity, t, mid_year)
    if terminal.get("method") == "gordon":
        g = float(terminal["g"])
        fcf_next = fcfe[-1] * (1.0 + g)
        tv = terminal_value_gordon(fcf_next, cost_of_equity, g)
    elif terminal.get("method") == "exit_multiple":
        multiple = float(terminal["multiple"])
        metric = float(terminal["metric"])
        tv = terminal_value_exit_multiple(metric, multiple)
    else: raise ValueError("terminal.method must be 'gordon' or 'exit_multiple'")
    N = len(fcfe)
    pv_terminal = tv * _df(cost_of_equity, N, mid_year)
    equity_value = pv_projection + pv_terminal
    out: Dict[str, Any] = {
        "equity_value": float(equity_value), "pv_projection": float(pv_projection),
        "terminal_value": float(tv), "pv_terminal": float(pv_terminal),
        "assumptions": {
            "cost_of_equity": float(cost_of_equity), "mid_year": bool(mid_year),
            "terminal_method": terminal.get("method"),
            **({"g": float(terminal["g"])} if terminal.get("method") == "gordon" else {}),
            **({"multiple": float(terminal["multiple"]), "metric": float(terminal["metric"])}
               if terminal.get("method") == "exit_multiple" else {}),
        },
    }
    if shares_outstanding and shares_outstanding > 0:
        out["equity_value_per_share"] = float(equity_value / shares_outstanding)
    return out

_VALUATION_OPS = {
    "npv": npv, "irr": irr, "terminal_value_gordon": terminal_value_gordon,
    "terminal_value_exit_multiple": terminal_value_exit_multiple, "dcf_fcff": dcf_fcff,
    "dcf_fcfe": dcf_fcfe, "wacc": wacc, "cost_of_equity_capm": cost_of_equity_capm,
}


class NPVInputs(BaseModel):
    operation: Literal["npv"]
    cash_flows: List[float] = Field(min_length=1)
    discount_rate: float
    mid_year: bool = False
    include_t0: bool = False

class IRRInputs(BaseModel):
    operation: Literal["irr"]
    cash_flows: List[float] = Field(min_length=2)

class TVGordonInputs(BaseModel):
    operation: Literal["terminal_value_gordon"]
    fcf_next_year: float
    rate: float
    g: float
    @model_validator(mode='after')
    def check_rate_gt_g(self) -> 'TVGordonInputs':
        if self.rate <= self.g: raise ValueError("rate must be greater than g")
        return self

class TVExitMultipleInputs(BaseModel):
    operation: Literal["terminal_value_exit_multiple"]
    metric_value: float
    multiple: float

class WACCInputs(BaseModel):
    operation: Literal["wacc"]
    cost_of_equity: float
    cost_of_debt_pre_tax: float
    tax_rate: float
    equity_value: float
    debt_value: float

class CAPMInputs(BaseModel):
    operation: Literal["cost_of_equity_capm"]
    risk_free: float
    beta: float
    market_premium: float

class TerminalGordonSchema(BaseModel):
    method: Literal["gordon"]
    g: float

class TerminalExitMultipleSchema(BaseModel):
    method: Literal["exit_multiple"]
    multiple: float
    metric: float

AnyTerminalInput = Union[TerminalGordonSchema, TerminalExitMultipleSchema]

class DCFFCFFInputs(BaseModel):
    operation: Literal["dcf_fcff"]
    fcff: List[float] = Field(min_length=1)
    wacc_rate: float
    terminal: AnyTerminalInput
    mid_year: bool = True
    net_debt: float = 0.0
    minority_interest: float = 0.0
    cash_and_investments: float = 0.0
    shares_outstanding: Optional[float] = None

class DCFFCFEInputs(BaseModel):
    operation: Literal["dcf_fcfe"]
    fcfe: List[float] = Field(min_length=1)
    cost_of_equity: float
    terminal: AnyTerminalInput
    mid_year: bool = True
    shares_outstanding: Optional[float] = None


class AnyValuationInput(RootModel):
    root: Union[
        NPVInputs, IRRInputs, TVGordonInputs, TVExitMultipleInputs,
        WACCInputs, CAPMInputs, DCFFCFFInputs, DCFFCFEInputs
    ]


def valuation_tool(tool_input: AnyValuationInput) -> Dict[str, Any]: 
  
    """
    Perform a specific valuation calculation.
    Input must be a JSON object matching the schema for the specific operation.
    """
    try:
       
        actual_input = tool_input.root 
        
        operation = actual_input.operation
        fn_to_call = _VALUATION_OPS.get(operation)
        if fn_to_call is None:
            return {"error": f"Unsupported operation: {operation}"}
        
        input_args = actual_input.model_dump(exclude={"operation"})
        
        if operation in ("dcf_fcff", "dcf_fcfe"):
            input_args['terminal'] = actual_input.terminal.model_dump()
       
        result = fn_to_call(**input_args)

        if operation in ("dcf_fcff", "dcf_fcfe"):
            return result
        
        return {
            "result": result,
            "operation": operation,
            "inputs": input_args
        }

    except (ValueError, TypeError) as e:
        
        return {"error": str(e), "operation": getattr(tool_input.root, 'operation', 'unknown')}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}