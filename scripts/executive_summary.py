import requests
import json
import os
import sys

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

# Get API key
API_KEY = os.getenv("GROQ_API_KEY")

# Validate API key exists
if not API_KEY:
    print("ERROR: API key not found. Please set GROQ_API_KEY or OPENROUTER_API_KEY environment variable.")
    sys.exit(1)

# Load input data
input_file = "output/insight_summary.json"
if not os.path.exists(input_file):
    print(f"ERROR: Input file {input_file} not found.")
    sys.exit(1)

try:
    with open(input_file) as f:
        data = json.load(f)
except json.JSONDecodeError as e:
    print(f"ERROR: Failed to parse JSON from {input_file}: {e}")
    sys.exit(1)

prompt = f"""
You are a Chief Experience Officer AI assistant.

Based on this weekly sentiment data:
{json.dumps(data, indent=2)}

Provide:
- 3 Key Insights
- 2 Risks
- 3 Strategic Recommendations

Keep it concise and executive-ready.
"""

try:
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "openai/gpt-oss-120b",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        },
        timeout=30
    )

    response.raise_for_status()
    result = response.json()["choices"][0]["message"]["content"]

    print("✓ Executive Summary Generated Successfully")
    print("\n" + "="*50)
    print(result)
    print("="*50 + "\n")

    with open("output/executive_summary.txt", "w") as f:
        f.write(result)
    
    print("✓ Summary saved to output/executive_summary.txt")

except requests.exceptions.HTTPError as e:
    print(f"ERROR: API request failed with status {e.response.status_code}")
    sys.exit(1)
except requests.exceptions.RequestException as e:
    print(f"ERROR: Request failed: {e}")
    sys.exit(1)
except (KeyError, IndexError) as e:
    print(f"ERROR: Failed to parse API response: {e}")
    sys.exit(1)