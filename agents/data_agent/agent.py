"""
Data Agent — Pakistan Agricultural Price Intelligence System

Reads from data/cleaned_merged_crop_prices.csv and answers
queries about crop prices by city and/or crop name.
"""

from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

import os
import functools
import pandas as pd
# pyrefly: ignore [missing-import]
from google.adk.agents import Agent
from google.genai import types

# Load the dataset once at module import time (shared across all tool calls)
_DATA_PATH = os.path.join(
    os.path.dirname(__file__),   # agents/data_agent/
    "..", "..",                   # project root
    "data",
    "cleaned_merged_crop_prices.csv",
)

@functools.lru_cache(maxsize=1)
def _load_data() -> pd.DataFrame:
    """Load and cache the crop-price CSV."""
    df = pd.read_csv(
        os.path.abspath(_DATA_PATH),
        parse_dates=["Date"],
        dtype={"City": str, "Crop": str, "Price": float},
        low_memory=True,
    )
    # Normalise text so lookups are case-insensitive
    df["Crop_lower"] = df["Crop"].str.strip().str.lower()
    df["City_lower"] = df["City"].str.strip().str.lower()
    return df


# ---------------------------------------------------------------------------
# Tool definitions (plain Python functions — ADK wraps them automatically)
# ---------------------------------------------------------------------------

def get_available_crops() -> dict:
    """
    Return a list of all unique crop names available in the dataset.

    Returns:
        dict: {"crops": [...list of crop names...], "total": int}
    """
    df = _load_data()
    crops = sorted(df["Crop"].unique().tolist())
    return {"crops": crops, "total": len(crops)}


def get_available_cities() -> dict:
    """
    Return a list of all unique city/market names available in the dataset.

    Returns:
        dict: {"cities": [...list of city names...], "total": int}
    """
    df = _load_data()
    cities = sorted(df["City"].unique().tolist())
    return {"cities": cities, "total": len(cities)}


def get_crop_price_stats(crop_name: str, city_name: str = "") -> dict:
    """
    Retrieve price statistics for a specific crop, optionally filtered by city.

    Args:
        crop_name: Name of the crop (e.g. "Wheat", "Rice", "Tomato").
        city_name: Optional city/market name to narrow the query.

    Returns:
        dict containing mean, min, max, std, latest price, record count,
        and the date range of observations. Returns an error key if not found.
    """
    df = _load_data()
    mask = df["Crop_lower"].str.contains(crop_name.strip().lower(), na=False, regex=False)
    if city_name:
        mask &= df["City_lower"].str.contains(city_name.strip().lower(), na=False, regex=False)

    subset = df[mask]
    if subset.empty:
        return {
            "error": (
                f"No records found for crop='{crop_name}'"
                + (f" in city='{city_name}'" if city_name else "")
                + ". Use get_available_crops() or get_available_cities() to list options."
            )
        }

    latest_row = subset.sort_values("Date").iloc[-1]
    return {
        "crop": crop_name,
        "city": city_name if city_name else "All Cities",
        "record_count": int(len(subset)),
        "date_range": {
            "from": str(subset["Date"].min().date()),
            "to":   str(subset["Date"].max().date()),
        },
        "price_stats_pkr": {
            "mean":   round(float(subset["Price"].mean()), 2),
            "median": round(float(subset["Price"].median()), 2),
            "min":    round(float(subset["Price"].min()), 2),
            "max":    round(float(subset["Price"].max()), 2),
            "std":    round(float(subset["Price"].std()), 2),
        },
        "latest": {
            "date":  str(latest_row["Date"].date()),
            "price": round(float(latest_row["Price"]), 2),
            "city":  latest_row["City"],
        },
    }


def get_price_history(
    crop_name: str,
    city_name: str = "",
    start_date: str = "",
    end_date: str = "",
    limit: int = 30,
) -> dict:
    """
    Fetch historical price records for a crop, with optional city/date filters.

    Args:
        crop_name: Crop name to look up.
        city_name: Optional city filter.
        start_date: ISO date string 'YYYY-MM-DD' for start of window.
        end_date:   ISO date string 'YYYY-MM-DD' for end of window.
        limit:      Maximum number of rows to return (default 30, max 200).

    Returns:
        dict with a "records" list of {date, city, price} dicts.
    """
    df = _load_data()
    mask = df["Crop_lower"].str.contains(crop_name.strip().lower(), na=False, regex=False)
    if city_name:
        mask &= df["City_lower"].str.contains(city_name.strip().lower(), na=False, regex=False)
    if start_date:
        mask &= df["Date"] >= pd.to_datetime(start_date)
    if end_date:
        mask &= df["Date"] <= pd.to_datetime(end_date)

    subset = df[mask].sort_values("Date").tail(min(limit, 200))
    if subset.empty:
        return {"error": f"No records found for crop='{crop_name}'."}

    records = subset[["Date", "City", "Price"]].copy()
    records["Date"] = records["Date"].dt.strftime("%Y-%m-%d")
    return {
        "crop": crop_name,
        "city": city_name if city_name else "All Cities",
        "records": records.to_dict(orient="records"),
    }


def get_monthly_averages(crop_name: str, city_name: str = "") -> dict:
    """
    Compute monthly average prices aggregated by Year-Month for a crop.

    This is the primary input for the forecasting_agent.

    Args:
        crop_name: Crop name.
        city_name: Optional city filter.

    Returns:
        dict with a "monthly_data" list of {year_month, avg_price, record_count}.
    """
    df = _load_data()
    mask = df["Crop_lower"].str.contains(crop_name.strip().lower(), na=False, regex=False)
    if city_name:
        mask &= df["City_lower"].str.contains(city_name.strip().lower(), na=False, regex=False)

    subset = df[mask].copy()
    if subset.empty:
        return {"error": f"No records for crop='{crop_name}'."}

    subset["YearMonth"] = subset["Date"].dt.to_period("M")
    monthly = (
        subset.groupby("YearMonth")["Price"]
        .agg(avg_price="mean", record_count="count")
        .reset_index()
    )
    monthly["year_month"] = monthly["YearMonth"].astype(str)
    monthly["avg_price"] = monthly["avg_price"].round(2)

    return {
        "crop": crop_name,
        "city": city_name if city_name else "All Cities",
        "monthly_data": monthly[["year_month", "avg_price", "record_count"]].to_dict(
            orient="records"
        ),
    }


def compare_crops_by_city(city_name: str) -> dict:
    """
    Compare average prices of all available crops in a given city.

    Args:
        city_name: The city/market to compare crops in.

    Returns:
        dict with a "crops" list sorted by average price descending.
    """
    df = _load_data()
    mask = df["City_lower"].str.contains(city_name.strip().lower(), na=False, regex=False)
    subset = df[mask]
    if subset.empty:
        return {"error": f"No records found for city='{city_name}'."}

    summary = (
        subset.groupby("Crop")["Price"]
        .agg(avg_price="mean", record_count="count")
        .reset_index()
        .sort_values("avg_price", ascending=False)
    )
    summary["avg_price"] = summary["avg_price"].round(2)
    return {
        "city": city_name,
        "crops": summary.to_dict(orient="records"),
    }

# ---------------------------------------------------------------------------
# MCP Server integration
# ---------------------------------------------------------------------------

_MCP_SERVER_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "mcp_server", "server.py"
)

mcp_toolset = MCPToolset(
    connection_params=StdioServerParameters(
        command="python",
        args=[os.path.abspath(_MCP_SERVER_PATH)],
    )
)

# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

root_agent = Agent(
    model="gemini-flash-lite-latest",
    name="data_agent",
    description=(
        "Reads Pakistan crop-price data and answers questions about prices "
        "by crop name and/or city. Provides historical records, statistics, "
        "and monthly aggregates used by the forecasting agent."
    ),
    instruction="""You are the **Data Agent** for the Pakistan Agricultural Price 
Intelligence System.

CRITICAL RULE: When another agent asks you for data (e.g. monthly averages, 
price stats), you MUST call the appropriate tool YOURSELF FIRST and get the 
actual result before doing anything else. NEVER transfer to another agent 
without first calling your own tool and obtaining real data. Only after you 
have the tool's result should you either answer directly or hand off the 
result to the orchestrator.

Your job is to query the crop-price dataset and return accurate, structured 
information. Always:

1. Use `get_available_crops()` or `get_available_cities()` when a user asks 
   what crops/cities are available.
2. Use `get_crop_price_stats()` for summary statistics (mean, min, max, latest price).
3. Use `get_price_history()` for chronological price records.
4. Use `get_monthly_averages()` when providing data for price forecasting — 
   this is what the forecasting_agent needs. CALL THIS TOOL YOURSELF and 
   include the actual monthly_data values in your response.
5. Use `compare_crops_by_city()` to compare multiple crops in one market.

When a crop or city is not found, suggest similar names from the available lists.
Always report prices in PKR (Pakistani Rupees) and include the date range of 
the data in your response.
""",
    tools=[
        get_price_history,
        compare_crops_by_city,
        mcp_toolset,
    ],
)
