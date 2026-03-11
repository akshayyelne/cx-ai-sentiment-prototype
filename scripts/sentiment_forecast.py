import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta

# Load curated data
df = pd.read_csv("processed/curated_feedback.csv")

# Ensure date column is datetime
df["date"] = pd.to_datetime(df["date"])

# Aggregate daily average sentiment
daily = df.groupby("date")["sentiment_score"].mean().reset_index()

# Create numeric time index
daily["time_index"] = np.arange(len(daily))

# Train simple regression model
X = daily[["time_index"]]
y = daily["sentiment_score"]

model = LinearRegression()
model.fit(X, y)

# Forecast next 7 days
future_days = 7
last_index = daily["time_index"].max()

future_indices = np.arange(last_index + 1, last_index + 1 + future_days)
future_dates = [daily["date"].max() + timedelta(days=i+1) for i in range(future_days)]

forecast = model.predict(future_indices.reshape(-1, 1))

forecast_df = pd.DataFrame({
    "date": future_dates,
    "forecast_sentiment": forecast
})

# Save forecast
forecast_df.to_csv("output/sentiment_forecast.csv", index=False)

print("✓ Forecast generated and saved to output/sentiment_forecast.csv")