import pandas as pd
import requests
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# CONFIGURATION
# ==============================

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "openai/gpt-oss-120b"

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    print("❌ GROQ_API_KEY not found.")
    sys.exit(1)

# ==============================
# LOAD DATA
# ==============================

try:
    df = pd.read_csv("processed/curated_feedback.csv")
except Exception as e:
    print(f"❌ Failed to load sentiment data: {e}")
    sys.exit(1)

# Optional forecast data
try:
    forecast_df = pd.read_csv("output/sentiment_actual_vs_forecast.csv")
except:
    forecast_df = None

# Combine text for retrieval
df["combined_text"] = (
    df["source"].astype(str)
    + " | "
    + df["feedback"].astype(str)
    + " | "
    + df["sentiment_label"].astype(str)
)

# ==============================
# VECTOR RETRIEVAL SETUP
# ==============================

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined_text"])


def retrieve_relevant_context(question, top_n=5):
    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, tfidf_matrix)
    top_indices = similarities.argsort()[0][-top_n:]
    return df.iloc[top_indices][["source", "feedback", "sentiment_label"]]


def include_forecast_context(question):
    keywords = ["forecast", "predict", "next week", "next month", "future", "trend"]
    return any(keyword in question.lower() for keyword in keywords)


# ==============================
# LLM CALL
# ==============================

def ask_llm(question, context_df):

    context_text = context_df.to_string(index=False)

    forecast_text = ""
    if forecast_df is not None and include_forecast_context(question):
        forecast_tail = forecast_df.tail(7)
        forecast_text = "\nForecast Data:\n" + forecast_tail.to_string(index=False)

    prompt = f"""
You are a Chief Experience Officer AI assistant.

Use ONLY the provided data. Do not invent information.

Historical Sentiment Data:
{context_text}

{forecast_text}

Question:
{question}

Provide:
- Direct answer
- Key drivers
- Risk outlook (if relevant)
- Recommended action
Keep response concise and executive-ready.
"""

    try:
        response = requests.post(
            GROQ_ENDPOINT,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3
            },
            timeout=30
        )

        if response.status_code != 200:
            return f"⚠️ API Error {response.status_code}: {response.text}"

        data = response.json()

        if "choices" not in data:
            return f"⚠️ Unexpected API response: {data}"

        return data["choices"][0]["message"]["content"]

    except requests.exceptions.RequestException as e:
        return f"⚠️ Request failed: {e}"


# ==============================
# INTERACTIVE LOOP
# ==============================

print("\n🔵 CX Conversational Assistant Ready")
print("Type 'exit' to quit.\n")

while True:
    question = input("Ask CX Assistant: ")

    if question.lower() == "exit":
        print("👋 Exiting CX Assistant.")
        break

    if not question.strip():
        continue

    context = retrieve_relevant_context(question)
    answer = ask_llm(question, context)

    print("\n" + "=" * 60)
    print(answer)
    print("=" * 60 + "\n")