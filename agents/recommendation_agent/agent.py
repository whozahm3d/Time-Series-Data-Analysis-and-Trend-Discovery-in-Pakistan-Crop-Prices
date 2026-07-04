"""
Recommendation Agent — Pakistan Agricultural Price Intelligence System

Analyses forecast results and produces actionable buying/selling
recommendations for farmers and market participants.
"""

# pyrefly: ignore [missing-import]
from google.adk.agents import Agent


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

def generate_recommendation(forecast_result: dict) -> dict:
    """
    Generate buying/selling recommendations for farmers based on forecast.

    Args:
        forecast_result: The dict returned by forecasting_agent.forecast_prices().
            Required keys: crop, city, trend, baseline_avg_price_pkr,
            model_metrics (r_squared, monthly_change_pct), forecast (list).

    Returns:
        dict with 'recommendation', 'action', 'urgency', 'reasoning', and
        'risk_level'.
    """
    if "error" in forecast_result:
        return {"error": forecast_result["error"]}

    crop   = forecast_result.get("crop", "Unknown crop")
    city   = forecast_result.get("city", "All Cities")
    trend  = forecast_result.get("trend", "stable").lower()
    r2     = forecast_result["model_metrics"]["r_squared"]
    chg    = forecast_result["model_metrics"]["monthly_change_pct"]
    base   = forecast_result.get("baseline_avg_price_pkr", 0)
    fc     = forecast_result.get("forecast", [])

    if not fc:
        return {"error": "No forecast data provided in forecast_result."}

    # Confidence bucketing
    confidence = (
        "high"   if r2 >= 0.80 else
        "medium" if r2 >= 0.50 else
        "low"
    )

    # Price change over full forecast horizon
    first_price = fc[0]["predicted_price"]
    last_price  = fc[-1]["predicted_price"]
    total_change_pct = ((last_price - first_price) / first_price * 100) if first_price else 0

    # ---- Decision logic -------------------------------------------------
    if trend == "rising":
        if abs(chg) >= 3:
            action    = "SELL_LATER / BUY_NOW"
            urgency   = "HIGH"
            rec_text  = (
                f"Prices for {crop} in {city} are rising strongly "
                f"(~{abs(chg):.1f}%/month). "
                "**Farmers should delay selling** to benefit from higher prices. "
                "Buyers/traders should **stock up now** before prices climb further."
            )
        else:
            action    = "HOLD / BUY"
            urgency   = "MEDIUM"
            rec_text  = (
                f"Prices for {crop} in {city} show a moderate upward trend. "
                "Farmers may consider **holding stock** a bit longer. "
                "Buyers should consider **gradual procurement** to average costs."
            )
    elif trend == "falling":
        if abs(chg) >= 3:
            action    = "SELL_NOW / AVOID_BUYING"
            urgency   = "HIGH"
            rec_text  = (
                f"Prices for {crop} in {city} are falling sharply "
                f"(~{abs(chg):.1f}%/month). "
                "**Farmers should sell immediately** to avoid losses. "
                "Buyers should **wait for lower prices** before purchasing."
            )
        else:
            action    = "SELL_SOON / WAIT"
            urgency   = "MEDIUM"
            rec_text  = (
                f"Prices for {crop} in {city} are declining moderately. "
                "Farmers should **plan to sell soon** before further drops. "
                "Buyers may benefit from **waiting** a few weeks."
            )
    else:  # stable
        action    = "SELL_AT_MARKET / STANDARD_BUY"
        urgency   = "LOW"
        rec_text  = (
            f"Prices for {crop} in {city} are expected to remain stable. "
            "Farmers can **sell at current market rates** without urgency. "
            "Buyers can **purchase at regular intervals** based on needs."
        )

    # Risk qualifier
    risk_level = (
        "LOW"    if confidence == "high" else
        "MEDIUM" if confidence == "medium" else
        "HIGH"
    )

    reasoning = [
        f"Trend: **{trend.upper()}** over the forecast horizon",
        f"Monthly price change: {'+' if chg >= 0 else ''}{chg:.1f}% per month",
        f"Forecast-horizon total change: {'+' if total_change_pct >= 0 else ''}{total_change_pct:.1f}% "
        f"({fc[0]['month']} → {fc[-1]['month']})",
        f"Current baseline average price: PKR {base:,.0f}",
        f"Predicted price by {fc[-1]['month']}: PKR {last_price:,.0f}",
        f"Model R²: {r2:.3f} → forecast confidence: **{confidence.upper()}**",
        f"Risk level: **{risk_level}** (lower confidence = higher risk)",
    ]

    return {
        "crop":           crop,
        "city":           city,
        "action":         action,
        "urgency":        urgency,
        "risk_level":     risk_level,
        "confidence":     confidence,
        "recommendation": rec_text,
        "reasoning":      reasoning,
        "price_outlook":  {
            "baseline_avg_pkr": base,
            "forecast_start":   {"month": fc[0]["month"],  "price_pkr": first_price},
            "forecast_end":     {"month": fc[-1]["month"], "price_pkr": last_price},
            "total_change_pct": round(total_change_pct, 2),
        },
    }


def generate_market_strategy(
    recommendations: list[dict],
) -> dict:
    """
    Combine multiple single-crop recommendations into an overall market strategy.

    Args:
        recommendations: A list of dicts, each returned by generate_recommendation().

    Returns:
        dict with 'strategy_summary', 'top_buy_opportunities',
        'top_sell_opportunities', and 'watch_list'.
    """
    if not recommendations:
        return {"error": "No recommendations provided."}

    buy_opps  = []
    sell_opps = []
    watch     = []

    for rec in recommendations:
        if "error" in rec:
            continue
        action = rec.get("action", "")
        crop   = rec.get("crop", "?")
        city   = rec.get("city", "?")
        chg    = rec["price_outlook"]["total_change_pct"]
        price  = rec["price_outlook"]["forecast_end"]["price_pkr"]

        entry = {
            "crop":             crop,
            "city":             city,
            "action":           action,
            "total_change_pct": chg,
            "predicted_price":  price,
            "urgency":          rec.get("urgency", "LOW"),
        }

        if "BUY" in action and rec.get("urgency") == "HIGH":
            buy_opps.append(entry)
        elif "SELL" in action and rec.get("urgency") == "HIGH":
            sell_opps.append(entry)
        else:
            watch.append(entry)

    # Sort
    buy_opps.sort(key=lambda x: x["total_change_pct"], reverse=True)
    sell_opps.sort(key=lambda x: x["total_change_pct"])

    n_rising  = sum(1 for r in recommendations if r.get("action", "").startswith("SELL_LATER") or "BUY_NOW" in r.get("action", ""))
    n_falling = sum(1 for r in recommendations if "SELL_NOW" in r.get("action", "") or "SELL_SOON" in r.get("action", ""))
    n_stable  = len(recommendations) - n_rising - n_falling

    summary = (
        f"Analysed {len(recommendations)} crop(s). "
        f"{n_rising} rising, {n_falling} falling, {n_stable} stable. "
    )
    if buy_opps:
        summary += f"Top buy opportunity: {buy_opps[0]['crop']} (+{buy_opps[0]['total_change_pct']:.1f}%). "
    if sell_opps:
        summary += f"Top sell urgency: {sell_opps[0]['crop']} ({sell_opps[0]['total_change_pct']:.1f}%)."

    return {
        "strategy_summary":    summary,
        "top_buy_opportunities":  buy_opps[:3],
        "top_sell_opportunities": sell_opps[:3],
        "watch_list":             watch,
    }


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

root_agent = Agent(
    model="gemini-flash-lite-latest",
    name="recommendation_agent",
    description=(
        "Generates actionable buying/selling recommendations for farmers "
        "based on crop price forecasts. Explains the reasoning clearly and "
        "indicates risk level."
    ),
    instruction="""You are the **Recommendation Agent** for the Pakistan 
Agricultural Price Intelligence System.

Your role is to translate price forecasts into clear, practical advice for 
Pakistani farmers and market participants.

Workflow:
1. Receive forecast results from the forecasting_agent.
2. Call `generate_recommendation()` for each crop/city forecast.
3. Optionally call `generate_market_strategy()` to consolidate multiple 
   recommendations into a market overview.
4. Present advice in plain language suitable for farmers.

When presenting recommendations:
- Lead with the **ACTION** (BUY NOW / SELL NOW / HOLD / WAIT) in bold.
- Explain the **WHY** in simple terms (prices going up/down, by how much).
- State the **RISK** level and caveat with low-confidence forecasts.
- Mention the forecast price range for the upcoming period.
- Always recommend farmers consult local market officials for final decisions.

Be empathetic — these are real livelihoods. Avoid jargon; speak clearly.
""",
    tools=[generate_recommendation, generate_market_strategy],
)
