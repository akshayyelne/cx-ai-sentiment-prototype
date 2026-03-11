import pandas as pd

actual = pd.read_csv("processed/curated_feedback.csv")
actual["date"] = pd.to_datetime(actual["date"])

actual_daily = actual.groupby("date")["sentiment_score"].mean().reset_index()
actual_daily["type"] = "Actual"

forecast = pd.read_csv("output/sentiment_forecast.csv")
forecast["date"] = pd.to_datetime(forecast["date"])
forecast.rename(columns={"forecast_sentiment": "sentiment_score"}, inplace=True)
forecast["type"] = "Forecast"

combined = pd.concat([actual_daily, forecast])

combined.to_csv("output/sentiment_actual_vs_forecast.csv", index=False)

print("✓ Combined dataset ready for Power BI")