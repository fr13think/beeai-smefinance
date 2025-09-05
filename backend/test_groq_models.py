from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Test different models
models_to_test = [
    "llama3-70b-8192",
    "llama3-8b-8192", 
    "llama-3.3-70b-versatile",
    "gemma-7b-it",
    "gemma2-9b-it"
]

print("Testing Groq Models Availability:\n")

for model in models_to_test:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        print(f"✅ {model}: AVAILABLE")
    except Exception as e:
        print(f"❌ {model}: {str(e)[:50]}...")

print("\nRecommended: Use llama3-8b-8192 for speed or llama3-70b-8192 for quality")
