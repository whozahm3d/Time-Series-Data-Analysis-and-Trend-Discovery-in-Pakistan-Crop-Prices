"""
MCP Server — Pakistan Agricultural Price Intelligence System

Exposes crop price data as MCP tools over stdio, so any MCP-compatible
client (Claude Desktop, ADK agents, etc.) can query the dataset.

Run standalone for testing:
    python mcp_server/server.py
"""

import os
import functools
import pandas as pd
import sys

sys.path.append(os.path.dirname(__file__))
# pyrefly: ignore [missing-import]
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("agri-price-data")

_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "cleaned_merged_crop_prices.csv"
)


@functools.lru_cache(maxsize=1)
def _load_data() -> pd.DataFrame:
    df = pd.read_csv(
        os.path.abspath(_DATA_PATH),
        parse_dates=["Date"],
        dtype={"City": str, "Crop": str, "Price": float},
        low_memory=True,
    )
    df["Crop_lower"] = df["Crop"].str.strip().str.lower()
    df["City_lower"] = df["City"].str.strip().str.lower()
    return df


@mcp.tool()
def get_available_crops() -> dict:
    """Return all unique crop names available in the Pakistan crop price dataset."""
    df = _load_data()
    crops = sorted(df["Crop"].unique().tolist())
    return {"crops": crops, "total": len(crops)}


@mcp.tool()
def get_available_cities() -> dict:
    """Return all unique city/market names available in the dataset."""
    df = _load_data()
    cities = sorted(df["City"].unique().tolist())
    return {"cities": cities, "total": len(cities)}


@mcp.tool()
def get_crop_price_stats(crop_name: str, city_name: str = "") -> dict:
    """
    Get price statistics (mean, min, max, latest) for a crop, optionally
    filtered by city.

    Args:
        crop_name: Name of the crop (e.g. "Apple", "Wheat").
        city_name: Optional city/market name.
    """
    df = _load_data()
    mask = df["Crop_lower"].str.contains(crop_name.strip().lower(), na=False, regex=False)
    if city_name:
        mask &= df["City_lower"].str.contains(city_name.strip().lower(), na=False, regex=False)

    subset = df[mask]
    if subset.empty:
        return {"error": f"No records found for crop='{crop_name}'."}

    latest_row = subset.sort_values("Date").iloc[-1]
    return {
        "crop": crop_name,
        "city": city_name if city_name else "All Cities",
        "record_count": int(len(subset)),
        "price_stats_pkr": {
            "mean": round(float(subset["Price"].mean()), 2),
            "min": round(float(subset["Price"].min()), 2),
            "max": round(float(subset["Price"].max()), 2),
        },
        "latest": {
            "date": str(latest_row["Date"].date()),
            "price": round(float(latest_row["Price"]), 2),
        },
    }


@mcp.tool()
def get_monthly_averages(crop_name: str, city_name: str = "") -> dict:
    """
    Compute monthly average prices for a crop — used as input for forecasting.

    Args:
        crop_name: Crop name.
        city_name: Optional city filter.
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
        "monthly_data": monthly[["year_month", "avg_price", "record_count"]].to_dict(
            orient="records"
        ),
    }


if __name__ == "__main__":
    mcp.run()