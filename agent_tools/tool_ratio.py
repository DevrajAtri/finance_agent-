from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class RatioInput(BaseModel):
    """
    Input arguments for calculating financial ratios.
    Provide the 'ratio_name' and the necessary numeric values for that ratio.
    """
    ratio_name: str = Field(
        description="The name of the ratio (e.g. 'gross_margin', 'roe', 'current_ratio')."
    )
    
    
    revenue: Optional[float] = Field(default=None, description="Total Revenue")
    cost_of_goods_sold: Optional[float] = Field(default=None, description="Cost of Goods Sold (COGS)")
    operating_income: Optional[float] = Field(default=None, description="Operating Income (EBIT)")
    net_income: Optional[float] = Field(default=None, description="Net Income")
    
    # --- Balance Sheet Inputs ---
    total_assets: Optional[float] = Field(default=None, description="Total Assets")
    total_equity: Optional[float] = Field(default=None, description="Total Equity")
    current_assets: Optional[float] = Field(default=None, description="Current Assets")
    current_liabilities: Optional[float] = Field(default=None, description="Current Liabilities")
    inventory: Optional[float] = Field(default=None, description="Inventory")
    total_debt: Optional[float] = Field(default=None, description="Total Debt")
    long_term_debt: Optional[float] = Field(default=None, description="Long Term Debt")
    
    # --- Valuation/Share Inputs ---
    price_per_share: Optional[float] = Field(default=None, description="Share Price")
    earnings_per_share: Optional[float] = Field(default=None, description="EPS")
    book_value_per_share: Optional[float] = Field(default=None, description="Book Value per Share")
    weighted_average_shares: Optional[float] = Field(default=None, description="Weighted Average Shares")
    preferred_dividends: Optional[float] = Field(default=None, description="Preferred Dividends")
    
    # --- Cash Flow/Other ---
    free_cash_flow: Optional[float] = Field(default=None, description="Free Cash Flow")
    interest_expense: Optional[float] = Field(default=None, description="Interest Expense")


def ratio_calculator(tool_input: RatioInput) -> Dict[str, Any]:
    """
    Calculates the requested financial ratio based on the provided inputs.
    """
    try:
        name = tool_input.ratio_name.lower().replace(" ", "_").replace("-", "_")
        result = None
        
        def get(field): return getattr(tool_input, field) or 0.0

        # --- Profitability ---
        if name == "gross_margin":
            # (Revenue - COGS) / Revenue
            rev = get("revenue")
            if rev == 0: raise ValueError("Revenue cannot be zero")
            result = (rev - get("cost_of_goods_sold")) / rev

        elif name == "operating_margin":
            # Op Income / Revenue
            rev = get("revenue")
            if rev == 0: raise ValueError("Revenue cannot be zero")
            result = get("operating_income") / rev

        elif name == "net_margin":
            # Net Income / Revenue
            rev = get("revenue")
            if rev == 0: raise ValueError("Revenue cannot be zero")
            result = get("net_income") / rev

        elif name == "return_on_assets" or name == "roa":
            # Net Income / Total Assets
            assets = get("total_assets")
            if assets == 0: raise ValueError("Total Assets cannot be zero")
            result = get("net_income") / assets

        elif name == "return_on_equity" or name == "roe":
            # Net Income / Total Equity
            equity = get("total_equity")
            if equity == 0: raise ValueError("Total Equity cannot be zero")
            result = get("net_income") / equity

        # --- Liquidity ---
        elif name == "current_ratio":
            # Current Assets / Current Liabilities
            liab = get("current_liabilities")
            if liab == 0: raise ValueError("Current Liabilities cannot be zero")
            result = get("current_assets") / liab

        elif name == "quick_ratio":
            # (Current Assets - Inventory) / Current Liabilities
            liab = get("current_liabilities")
            if liab == 0: raise ValueError("Current Liabilities cannot be zero")
            result = (get("current_assets") - get("inventory")) / liab

        elif name == "debt_to_equity":
            # Total Debt / Total Equity
            equity = get("total_equity")
            if equity == 0: raise ValueError("Total Equity cannot be zero")
            result = get("total_debt") / equity

        elif name == "interest_coverage":
            # Operating Income / Interest Expense
            interest = get("interest_expense")
            if interest == 0: raise ValueError("Interest Expense cannot be zero")
            result = get("operating_income") / interest

        
        elif name == "asset_turnover":
            # Revenue / Total Assets
            assets = get("total_assets")
            if assets == 0: raise ValueError("Total Assets cannot be zero")
            result = get("revenue") / assets

        elif name == "inventory_turnover":
            # COGS / Inventory
            inv = get("inventory")
            if inv == 0: raise ValueError("Inventory cannot be zero")
            result = get("cost_of_goods_sold") / inv

        # --- Valuation ---
        elif name == "earnings_per_share" or name == "eps":
            # (Net Income - Pref Div) / Shares
            shares = get("weighted_average_shares")
            if shares == 0: raise ValueError("Shares cannot be zero")
            result = (get("net_income") - get("preferred_dividends")) / shares

        elif name == "price_to_earnings" or name == "pe":
            # Price / EPS
            eps = get("earnings_per_share")
            if eps == 0: raise ValueError("EPS cannot be zero")
            result = get("price_per_share") / eps

        elif name == "price_to_book" or name == "pb":
            # Price / Book Value
            bv = get("book_value_per_share")
            if bv == 0: raise ValueError("Book Value cannot be zero")
            result = get("price_per_share") / bv

        else:
            return {"error": f"Unknown ratio: {name}"}

       
        return {
            "result": round(float(result), 4),
            "ratio_name": name,
            "inputs": tool_input.model_dump(exclude_none=True)
        }

    except Exception as e:
        return {
            "error": f"Calculation failed: {str(e)}",
            "ratio_name": tool_input.ratio_name
        }