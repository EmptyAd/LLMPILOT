# rag_utils.py
import json
import chromadb
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="rag_docs")

def load_docs_from_json(json_path="rag_docs.json"):
    with open(json_path) as f:
        docs = json.load(f)

    for i, doc in enumerate(docs):
        doc_id = f"doc_{i}"  # Auto-generate ID
        collection.add(
            documents=[doc["content"]],
            metadatas=[{"source": doc.get("source", f"unknown_{i}")}],
            ids=[doc_id]
        )


def retrieve_relevant_docs(query, top_k=3):
    embedding = embedding_model.encode([query])[0]
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    return [doc for doc in results["documents"][0]]
