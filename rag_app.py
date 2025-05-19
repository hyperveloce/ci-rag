print("Starting rag_app.py...")

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
import requests
import os
import json
from flask import Flask, request, jsonify
from langchain_text_splitters import RecursiveCharacterTextSplitter
print(f"Initialize vector store called.")
print(f"Current dir: {os.getcwd()}")
print(f"test_data.txt exists: {os.path.exists('test_data.txt')}")
print("Basic imports successful...")

PERSIST_DIRECTORY = "./qdrant_db"
TEST_DATA_PATH = "test_data.txt"

vector_store = None  # Initialize globally

def initialize_vector_store():
    global vector_store
    print("Initializing vector store...")
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    client = QdrantClient(host="qdrant_instance", port=6333)

    print("Indexing documents...")
    loader = TextLoader("test_data.txt")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    vector_store = Qdrant.from_documents(
        chunks, embeddings_model, path=PERSIST_DIRECTORY, collection_name="my_documents"
    )

    # Debug: Check vector store size or similar property here:
    # This depends on your vector_store implementation, example:
    print(f"Vector store initialized with {len(chunks)} chunks indexed.")


def call_ollama_api(prompt):
    url = f"{os.environ.get('OLLAMA_BASE_URL', 'http://ollama:11434')}/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "model": os.environ.get("OLLAMA_MODEL", "llama3.1")
    }

    response = requests.post(url, headers=headers, json=data, stream=True)
    response.raise_for_status()

    full_response = ""
    for line in response.iter_lines():
        if line:
            decoded = line.decode('utf-8')
            print("Raw chunk received:", repr(decoded))  # <- You can now see individual lines
            try:
                chunk = json.loads(decoded)
                full_response += chunk.get("response", "")
                if chunk.get("done", False):
                    break
            except json.JSONDecodeError as e:
                print("JSON decoding error:", e)
                continue

    return full_response


app = Flask(__name__)

def call_ollama_api(prompt):
    url = f"{os.environ.get('OLLAMA_BASE_URL', 'http://ollama:11434')}/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "model": os.environ.get("OLLAMA_MODEL", "llama3.1")
    }

    response = requests.post(url, headers=headers, json=data, stream=True)
    response.raise_for_status()

    full_response = ""
    for line in response.iter_lines():
        if line:
            decoded = line.decode('utf-8')
            print("Raw chunk received:", repr(decoded))  # Optional for debugging
            try:
                chunk = json.loads(decoded)
                full_response += chunk.get("response", "")
                if chunk.get("done", False):
                    break
            except json.JSONDecodeError as e:
                print("JSON decoding error:", e)
                continue

    return full_response

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' in request body"}), 400

        user_query = data['query']

        # Step 1: Retrieve relevant docs from vector store
        if vector_store is None:
            return jsonify({"error": "Vector store not initialized"}), 500

        results = vector_store.similarity_search(user_query, k=3)  # get top 3 relevant docs

        # Step 2: Combine results into context text
        context_text = "\n\n".join([doc.page_content for doc in results])

        # Step 3: Create prompt including context
        prompt_with_context = f"Here is some context:\n{context_text}\n\nAnswer this question based on the above context:\n{user_query}"

        # Step 4: Call Ollama with augmented prompt
        answer = call_ollama_api(prompt_with_context)

        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Calling initialize_vector_store() before starting Flask...")
    initialize_vector_store()
    app.run(host="0.0.0.0", port=8000)
