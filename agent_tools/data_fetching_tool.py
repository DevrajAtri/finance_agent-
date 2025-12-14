import yfinance as yf
import pandas_datareader.data as pdr
import pandas as pd
import datetime
from pydantic import BaseModel, Field, field_validator, RootModel
from typing import List, Dict, Any, Literal, Union, Optional
import io

#  1. PYDANTIC SCHEMAS 
class GetCurrentPriceInput(BaseModel):
    operation: Literal["get_current_price"]
    ticker: str = Field(description="The stock ticker (e.g., 'AAPL', 'MSFT').")

class GetHistoricalDataInput(BaseModel):
    operation: Literal["get_historical_data"]
    ticker: str = Field(description="The stock ticker.")
    period: str = Field("1y", description="Duration (e.g., '1mo', '1y').")
    interval: str = Field("1d", description="Frequency (e.g., '1d', '1wk').")

class GetCompanyInfoInput(BaseModel):
    operation: Literal["get_company_info"]
    ticker: str = Field(description="The stock ticker.")

class GetEconomicDataInput(BaseModel):
    operation: Literal["get_economic_data"]
    series_id: str = Field(description="FRED Series ID.")
    source: Literal["fred"] = Field("fred")
    start_date: Optional[str] = Field(None)
    end_date: Optional[str] = Field(None)

#  2. THE FIX: ROOT MODEL WRAPPER 
class AnyDataFetchingInput(RootModel):
    root: Union[
        GetCurrentPriceInput,
        GetHistoricalDataInput,
        GetCompanyInfoInput,
        GetEconomicDataInput
    ]

#  3. CORE IMPLEMENTATION FUNCTIONS  
def _get_current_price(ticker: str) -> Dict[str, Any]:
    stock = yf.Ticker(ticker)
    info = stock.info
    required_keys = {"currentPrice": "current_price", "previousClose": "previous_close", "open": "open", "dayHigh": "day_high", "dayLow": "day_low", "volume": "volume", "marketCap": "market_cap"}
    result_data = {"ticker": ticker}
    for key, new_key in required_keys.items():
        if key in info: result_data[new_key] = info[key]
    if "current_price" not in result_data:
        hist = stock.history(period="1d")
        if not hist.empty: result_data["current_price"] = hist['Close'].iloc[-1]
        else: raise ValueError("Could not retrieve current price.")
    return result_data

def _get_historical_data(ticker: str, period: str, interval: str) -> Dict[str, Any]:
    stock = yf.Ticker(ticker)
    hist_df = stock.history(period=period, interval=interval)
    if hist_df.empty: raise ValueError(f"No historical data found for {ticker}.")
    csv_buffer = io.StringIO()
    hist_df.to_csv(csv_buffer)
    return {"ticker": ticker, "period": period, "interval": interval, "rows": len(hist_df), "result_csv": csv_buffer.getvalue()}

def _get_company_info(ticker: str) -> Dict[str, Any]:
    stock = yf.Ticker(ticker)
    info = stock.info
    safe_keys = ["sector", "industry", "longName", "country", "website", "marketCap", "beta", "trailingPE", "forwardPE", "bookValue", "priceToBook", "dividendYield", "payoutRatio"]
    result_data = {"ticker": ticker}
    for key in safe_keys:
        if key in info and info[key] is not None: result_data[key] = info[key]
    if len(result_data) == 1: raise ValueError(f"Could not retrieve info for {ticker}.")
    return result_data

def _get_economic_data(series_id: str, source: str, start_date: Optional[str], end_date: Optional[str]) -> Dict[str, Any]:
    start = datetime.datetime.strptime(start_date, '%Y-%m-%d') if start_date else datetime.datetime(2020, 1, 1)
    end = datetime.datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.datetime.now()
    df = pdr.DataReader(series_id, source, start, end)
    if df.empty: raise ValueError(f"No data found for {series_id}.")
    recent_value = df.iloc[-1].to_dict()
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer)
    return {"series_id": series_id, "source": source, "most_recent_data": recent_value, "rows": len(df), "result_csv": csv_buffer.getvalue()}

# --- 4. REFACTORED TOOL ENTRYPOINT ---
def data_fetching_tool(tool_input: AnyDataFetchingInput) -> Dict[str, Any]:
    
    """
    Fetches external data. Input must match the specific operation schema.
    """
    try:
        # 1. Unwrap the actual input
        actual_input = tool_input.root
        operation = actual_input.operation
        
        if operation == "get_current_price":
            result = _get_current_price(actual_input.ticker)
        elif operation == "get_historical_data":
            result = _get_historical_data(actual_input.ticker, actual_input.period, actual_input.interval)
        elif operation == "get_company_info":
            result = _get_company_info(actual_input.ticker)
        elif operation == "get_economic_data":
            result = _get_economic_data(actual_input.series_id, actual_input.source, actual_input.start_date, actual_input.end_date)
        else:
            return {"error": f"Unknown operation: {operation}"}

        return {"operation": operation, **result}

    except (ValueError, TypeError) as e:
        return {"error": str(e), "operation": getattr(tool_input.root, 'operation', 'unknown')}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}