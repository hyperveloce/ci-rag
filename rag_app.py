import os
import requests

ollama_base_url = os.environ.get("OLLAMA_BASE_URL")

def query_ollama(prompt):
    if not ollama_base_url:
        raise ValueError("OLLAMA_BASE_URL environment variable not set.")
    url = f"{ollama_base_url}/api/generate"
    data = {
        "prompt": prompt,
        "stream": False  # Set to True for streaming responses
    }
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama: {e}")
        return ""

def query_llm_with_rag(query, context):
    prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}"
    return query_ollama(prompt)

# ... rest of your RAG application logic ...
