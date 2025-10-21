import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # Match upload model
COLLECTION = "docs"
VECTOR_NAME = "content_vector"

def search(query: str, top_k: int = 5):
    print(f"Query: {query}")
    model = SentenceTransformer(MODEL_NAME)
    qvec = model.encode(query, normalize_embeddings=True).tolist()

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    results = client.search(
        collection_name=COLLECTION,
        query_vector=(VECTOR_NAME, qvec),  # Use tuple format for named vector
        limit=top_k,
        with_payload=True,
        score_threshold=None,
    )
    for i, r in enumerate(results, 1):
        payload = r.payload or {}
        print(f"\n#{i}  score={r.score:.4f}")
        print(f"Title : {payload.get('title')}")
        print(f"URL   : {payload.get('url')}")
        snippet = (payload.get('text') or "")[:240].replace("\n"," ")
        print(f"Text  : {snippet}…")
    if not results:
        print("No results. Try a different query or check collection name.")

if __name__ == "__main__":
    # Try a couple of domain queries you care about:
    search("Incorta SAML SSO setup", top_k=5)
    # search("materialized view refresh error", top_k=5)
