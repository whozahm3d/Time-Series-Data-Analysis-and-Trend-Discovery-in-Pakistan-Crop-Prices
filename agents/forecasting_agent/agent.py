"""
Forecasting Agent — Pakistan Agricultural Price Intelligence System

Takes monthly crop price data and forecasts future prices using
linear regression (with optional seasonal features).
"""

import math
from typing import Any
# pyrefly: ignore [missing-import]
from google.adk.agents import Agent


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

def forecast_prices(
    monthly_data: list[dict[str, Any]],
    crop_name: str,
    city_name: str = "All Cities",
    forecast_months: int = 6,
) -> dict:
    """
    Forecast future crop prices using linear regression on monthly averages.

    Args:
        monthly_data: List of dicts with keys 'year_month' (str 'YYYY-MM'),
                      'avg_price' (float), 'record_count' (int).
                      Typically produced by data_agent.get_monthly_averages().
        crop_name:    Name of the crop being forecasted.
        city_name:    City/market scope (for labelling).
        forecast_months: How many months ahead to forecast (1–24, default 6).

    Returns:
        dict with 'trend', 'model_metrics', and 'forecast' list of
        {month, predicted_price, lower_bound, upper_bound}.
    """
    if not monthly_data:
        return {"error": "monthly_data is empty. Please provide historical price data."}

    forecast_months = max(1, min(int(forecast_months), 24))

    # ---- Prepare X (time index) and y (price) ----------------------------
    prices = [float(row["avg_price"]) for row in monthly_data]
    n = len(prices)

    if n < 3:
        return {
            "error": (
                f"Only {n} monthly observations — need at least 3 to fit a model."
            )
        }

    x_vals = list(range(n))  # 0, 1, 2, …

    # ---- Linear regression (pure Python — no sklearn dependency) ----------
    x_mean = sum(x_vals) / n
    y_mean = sum(prices) / n

    ss_xy = sum((x_vals[i] - x_mean) * (prices[i] - y_mean) for i in range(n))
    ss_xx = sum((x_vals[i] - x_mean) ** 2 for i in range(n))

    if ss_xx == 0:
        return {"error": "All time indices are identical — cannot fit regression."}

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    # ---- Goodness-of-fit (R²) --------------------------------------------
    y_pred_train = [slope * x + intercept for x in x_vals]
    ss_res = sum((prices[i] - y_pred_train[i]) ** 2 for i in range(n))
    ss_tot = sum((prices[i] - y_mean) ** 2 for i in range(n))
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    # ---- Residual standard error (for confidence bands) ------------------
    if n > 2:
        mse = ss_res / (n - 2)
        residual_std = math.sqrt(mse)
    else:
        residual_std = 0.0

    # ---- Forecast --------------------------------------------------------
    # Parse the last year_month to generate future labels
    last_period = monthly_data[-1]["year_month"]  # e.g. "2022-12"
    last_year, last_month = int(last_period[:4]), int(last_period[5:7])

    forecasts = []
    for step in range(1, forecast_months + 1):
        x_future = n - 1 + step
        predicted = slope * x_future + intercept
        # 95% prediction interval: ±1.96 × residual_std (approximate)
        margin = 1.96 * residual_std * math.sqrt(
            1 + 1 / n + (x_future - x_mean) ** 2 / ss_xx
        )

        month_idx = (last_month - 1 + step) % 12
        year_offset = (last_month - 1 + step) // 12
        fut_month = month_idx + 1
        fut_year = last_year + year_offset

        forecasts.append(
            {
                "month": f"{fut_year}-{fut_month:02d}",
                "predicted_price": round(max(0.0, predicted), 2),
                "lower_bound":     round(max(0.0, predicted - margin), 2),
                "upper_bound":     round(predicted + margin, 2),
            }
        )

    # ---- Trend direction -------------------------------------------------
    if slope > 0.5:
        trend = "rising"
    elif slope < -0.5:
        trend = "falling"
    else:
        trend = "stable"

    monthly_change_pct = (slope / y_mean * 100) if y_mean else 0.0

    return {
        "crop": crop_name,
        "city": city_name,
        "model": "Linear Regression",
        "training_months": n,
        "model_metrics": {
            "r_squared":        round(r_squared, 4),
            "residual_std_pkr": round(residual_std, 2),
            "slope_per_month":  round(slope, 2),
            "monthly_change_pct": round(monthly_change_pct, 2),
        },
        "trend": trend,
        "baseline_avg_price_pkr": round(y_mean, 2),
        "forecast": forecasts,
    }


def summarise_forecast(forecast_result: dict) -> dict:
    """
    Produce a human-readable summary of a forecast result.

    Args:
        forecast_result: The dict returned by forecast_prices().

    Returns:
        dict with 'summary' (str), 'key_stats' (dict).
    """
    if "error" in forecast_result:
        return {"error": forecast_result["error"]}

    fc = forecast_result.get("forecast", [])
    if not fc:
        return {"error": "No forecast data in result."}

    first = fc[0]
    last  = fc[-1]
    trend = forecast_result.get("trend", "unknown")
    r2    = forecast_result["model_metrics"]["r_squared"]
    slope = forecast_result["model_metrics"]["slope_per_month"]
    chg   = forecast_result["model_metrics"]["monthly_change_pct"]
    crop  = forecast_result.get("crop", "Unknown")
    city  = forecast_result.get("city", "All Cities")

    confidence = (
        "high"   if r2 >= 0.80 else
        "medium" if r2 >= 0.50 else
        "low"
    )

    summary = (
        f"For **{crop}** in **{city}**, prices are on a **{trend}** trend "
        f"(~{abs(chg):.1f}% per month). "
        f"Forecast spans {len(fc)} months: "
        f"{first['month']} (PKR {first['predicted_price']:,.0f}) → "
        f"{last['month']} (PKR {last['predicted_price']:,.0f}). "
        f"Model R² = {r2:.3f} ({confidence} confidence)."
    )

    return {
        "summary": summary,
        "key_stats": {
            "trend":                 trend,
            "monthly_slope_pkr":     slope,
            "monthly_change_pct":    chg,
            "r_squared":             r2,
            "confidence":            confidence,
            "first_forecast_month":  first["month"],
            "first_predicted_price": first["predicted_price"],
            "last_forecast_month":   last["month"],
            "last_predicted_price":  last["predicted_price"],
        },
    }


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

root_agent = Agent(
    model="gemini-flash-lite-latest",
    name="forecasting_agent",
    description=(
        "Forecasts future crop prices using linear regression on monthly "
        "price data supplied by the data_agent. Returns forecasts with "
        "confidence intervals and trend analysis."
    ),
    instruction="""You are the **Forecasting Agent** for the Pakistan 
Agricultural Price Intelligence System.

Your responsibilities:
1. Accept monthly price data (typically from the data_agent's 
   `get_monthly_averages` tool output).
2. Call `forecast_prices()` to generate a linear-regression-based forecast 
   for the requested number of months ahead (default 6).
3. Call `summarise_forecast()` to create a concise human-readable summary 
   of the forecast result.
4. Pass the full forecast result to the recommendation_agent for actionable 
   advice to farmers.

Always report:
- Trend direction (rising / stable / falling)
- Monthly price change (PKR and %)
- Model confidence (R²)
- Predicted prices with 95% confidence bands for each future month

If the data is sparse (<12 months), warn the user that forecast accuracy 
may be limited.
""",
    tools=[forecast_prices, summarise_forecast],
)
