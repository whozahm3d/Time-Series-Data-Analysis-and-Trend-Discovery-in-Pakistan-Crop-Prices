"""End-to-end smoke test for the agri price multi-agent system."""

# pyrefly: ignore [missing-import]
from agents.data_agent.agent import (
    get_available_crops, get_crop_price_stats, get_monthly_averages
)
# pyrefly: ignore [missing-import]
from agents.forecasting_agent.agent import forecast_prices, summarise_forecast
# pyrefly: ignore [missing-import]
from agents.recommendation_agent.agent import generate_recommendation

# 1. Available crops
crops_result = get_available_crops()
print(f"Available crops ({crops_result['total']}), first 5: {crops_result['crops'][:5]}")

# 2. Price stats for first crop
first_crop = crops_result['crops'][0]
stats = get_crop_price_stats(first_crop)
print(f"\nStats for '{first_crop}':")
print(f"  Records: {stats['record_count']}")
print(f"  Mean price: PKR {stats['price_stats_pkr']['mean']:,.0f}")
print(f"  Date range: {stats['date_range']['from']} to {stats['date_range']['to']}")

# 3. Monthly averages
monthly = get_monthly_averages(first_crop)
n = len(monthly['monthly_data'])
print(f"\nMonthly data points: {n}")

# 4. Forecast 6 months ahead
forecast_result = forecast_prices(monthly['monthly_data'], first_crop, 'All Cities', 6)
print(f"\nForecast trend: {forecast_result['trend']}")
print(f"R-squared: {forecast_result['model_metrics']['r_squared']}")
print("First 3 forecast months:")
for m in forecast_result['forecast'][:3]:
    print(f"  {m['month']}: PKR {m['predicted_price']:,.0f} [{m['lower_bound']:,.0f} - {m['upper_bound']:,.0f}]")

# 5. Summary
summary = summarise_forecast(forecast_result)
print(f"\nSummary:\n  {summary['summary']}")

# 6. Recommendation
rec = generate_recommendation(forecast_result)
print(f"\nAction: {rec['action']}")
print(f"Urgency: {rec['urgency']}")
print(f"Risk Level: {rec['risk_level']}")
print(f"Recommendation: {rec['recommendation']}")

print("\n=== SMOKE TEST PASSED ===")
