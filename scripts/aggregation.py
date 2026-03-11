import pandas as pd
import json

df = pd.read_csv("processed/curated_feedback.csv")

summary = {
    "total_records": len(df),
    "positive_count": int((df["sentiment_label"] == "Positive").sum()),
    "negative_count": int((df["sentiment_label"] == "Negative").sum()),
    "neutral_count": int((df["sentiment_label"] == "Neutral").sum()),
    "average_sentiment_score": round(df["sentiment_score"].mean(), 3)
}

with open("output/insight_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("Aggregation complete.")
