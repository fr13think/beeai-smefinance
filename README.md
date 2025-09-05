pip cache purge
python -m pip install --upgrade pip setuptools
pip install -r requirements.txt

python -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt


# Start backend
python main.py

# Buka index.html di browser
# Frontend tetap sama seperti sebelumnya


# Pilihan model Llama di Groq:
models = [
    "llama-3.2-90b-text-preview",  # Largest, most capable
    "llama-3.2-11b-text-preview",  # Balanced
    "llama-3.2-3b-preview",         # Fastest
    "llama3-70b-8192",              # Stable version
    "llama3-8b-8192"                # Lightweight
]

# 1. Test backend terlebih dahulu
curl http://localhost:5000/api/health

# 2. Test individual agent
curl -X POST http://localhost:5000/api/rag-query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are SME financing options?"}'

# 3. Monitor response time di browser console
# Buka Developer Tools > Network tab untuk melihat actual response time
