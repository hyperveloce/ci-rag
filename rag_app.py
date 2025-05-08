from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient, models
import os

PERSIST_DIRECTORY = "./qdrant_db"

def initialize_vector_store():
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    client = QdrantClient(path=PERSIST_DIRECTORY)

    # Load the test data file
    loader = TextLoader("test_data.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    vector_store = Qdrant.from_documents(
        chunks, embeddings_model, path=PERSIST_DIRECTORY, collection_name="my_documents"
    )
    return vector_store

# ... rest of your RAG application using initialize_vector_store() ...
