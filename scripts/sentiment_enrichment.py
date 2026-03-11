import streamlit as st
import pandas as pd
import requests
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from io import BytesIO

# ======================================
# CONFIGURATION
# ======================================

st.set_page_config(page_title="CX AI Assistant", layout="wide")
st.title("💬 CX Sentiment Intelligence Assistant")

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "openai/gpt-oss-120b"

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    st.error("GROQ_API_KEY not found.")
    st.stop()

# ======================================
# PDF GENERATION
# ======================================

def generate_project_pdf():
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []

    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    heading = styles["Heading1"]

    elements.append(Paragraph("Product Specification Document", heading))
    elements.append(Spacer(1, 0.5 * inch))

    content = """
CX Sentiment Intelligence Assistant

Core Capabilities:
• Identify key sentiment drivers
• Detect emerging CX risk
• Forecast sentiment trajectory
• Recommend strategic actions
• Conversational executive AI interface
"""

    for line in content.split("\n"):
        elements.append(Paragraph(line, normal))
        elements.append(Spacer(1, 0.2 * inch))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ======================================
# LOAD DATA
# ======================================

@st.cache_data
def load_data():
    df = pd.read_csv("processed/curated_feedback.csv")
    df["combined_text"] = (
        df["source"].astype(str)
        + " | "
        + df["feedback"].astype(str)
        + " | "
        + df["sentiment_label"].astype(str)
    )
    return df

@st.cache_data
def load_forecast():
    try:
        return pd.read_csv("output/sentiment_actual_vs_forecast.csv")
    except:
        return None

@st.cache_resource
def get_vectorizer_and_matrix():
    df = load_data()
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["combined_text"])
    return vectorizer, tfidf_matrix

# ======================================
# RETRIEVAL + INTENT
# ======================================

def retrieve_context(question, df, vectorizer, tfidf_matrix, top_n=5):
    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, tfidf_matrix)
    top_indices = similarities.argsort()[0][-top_n:]
    return df.iloc[top_indices][["source", "feedback", "sentiment_label"]]

def include_forecast_context(question):
    keywords = [
        "forecast", "predict", "trend", "future",
        "next week", "next month", "impact",
        "consequence", "decline", "exposure"
    ]
    return any(word in question.lower() for word in keywords)

def detect_intent(question):
    q = question.lower()

    driver_keywords = ["driver", "cause", "reason", "why", "root"]
    risk_keywords = [
        "risk", "forecast", "predict", "trend",
        "future", "impact", "decline", "exposure",
        "consequence", "next"
    ]
    action_keywords = [
        "action", "recommend", "improve",
        "priority", "stabilise", "stabilize",
        "mitigate", "fix"
    ]

    if any(word in q for word in driver_keywords):
        return "drivers"
    elif any(word in q for word in risk_keywords):
        return "risk"
    elif any(word in q for word in action_keywords):
        return "action"
    else:
        return "strategic"

# ======================================
# LLM CALL
# ======================================

def ask_llm(question, context_df, forecast_df):

    intent = detect_intent(question)

    if intent == "drivers":
        instruction = "Focus specifically on key drivers of sentiment."
    elif intent == "risk":
        instruction = "Focus on risk outlook and forecast implications."
    elif intent == "action":
        instruction = "Focus on recommended strategic actions."
    else:
        instruction = "Provide executive-level strategic insight."

    context_text = context_df.to_string(index=False)

    forecast_text = ""
    if forecast_df is not None and include_forecast_context(question):
        forecast_text = "\nForecast Data:\n" + forecast_df.tail(7).to_string(index=False)

    prompt = f"""
You are a Chief Experience Officer AI assistant.

Use ONLY the provided data. Do not invent information.

Historical Sentiment Data:
{context_text}

{forecast_text}

Question:
{question}

{instruction}

Keep the response concise and executive-ready.
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

# ======================================
# CHAT SESSION
# ======================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": """👋 Hello, I’m your CX Intelligence Assistant.

Ask me about:
• Key drivers
• Risk outlook
• Recommended actions"""
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask about customer sentiment...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    df = load_data()
    forecast_df = load_forecast()
    vectorizer, tfidf_matrix = get_vectorizer_and_matrix()

    with st.chat_message("assistant"):
        with st.spinner("Analyzing CX intelligence..."):
            context = retrieve_context(question, df, vectorizer, tfidf_matrix)
            answer = ask_llm(question, context, forecast_df)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ======================================
# BOTTOM RIGHT DOWNLOAD BUTTON
# ======================================

st.markdown("---")

col1, col2, col3 = st.columns([6, 2, 2])

with col3:
    pdf_file = generate_project_pdf()
    st.download_button(
        label="📄 Download Product Specification",
        data=pdf_file,
        file_name="Product Specification Document.pdf",
        mime="application/pdf",
        use_container_width=True
    )