
import os
from typing import Dict, Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from context.user_context import user_context


# Global resources (loaded once at module level)
_embedding_model = None
_qdrant_client = None


def get_embedding_model():
    """Lazy load embedding model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2",
            device="cpu"
        )
    return _embedding_model


def get_qdrant_client():
    """Get or create Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        # Get credentials from context or environment
        ctx = user_context.get()
        qdrant_url = ctx.get("qdrant_url") or os.getenv("QDRANT_URL")
        qdrant_api_key = ctx.get("qdrant_api_key") or os.getenv("QDRANT_API_KEY", "")

        _qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    return _qdrant_client


def search_knowledge_base(arguments: Dict[str, Any]) -> dict:
    """
    Search comprehensive Incorta documentation using vector similarity.
    
    This is your primary tool for accessing accurate, up-to-date Incorta information.
    Always use this tool before answering Incorta-specific questions.

    **Knowledge Base Sources:**
    - Incorta Community Documentation: Community-contributed guides and best practices
    - Official Incorta Documentation: Product documentation, features, and functionality
    - Incorta Support Documentation: Support articles, troubleshooting guides, and solutions
    
    **Use Cases:**
    - Answer questions about Incorta features, functionality, and best practices
    - Provide guidance on Incorta configuration and administration
    - Help troubleshoot Incorta-related issues
    - Explain Incorta concepts and terminology
    - Find specific documentation sources to cite in responses
    
    **Best Practices:**
    - Search before providing Incorta-specific answers
    - Use specific, focused queries for better results
    - Increase limit parameter for broader research
    - Cite sources from search results when answering questions

    Args:
        query (str): Search query - be specific for best results
        limit (int): Number of results to return (default: 5, increase for comprehensive research)

    Returns:
        dict: Search results with relevance scores, titles, URLs, and content snippets
    """
    query = arguments["query"]
    limit = arguments.get("limit", 5)

    # Get clients
    embedding_model = get_embedding_model()
    qdrant_client = get_qdrant_client()

    # Encode query
    query_vector = embedding_model.encode([query])[0]

    # Search Qdrant
    search_result = qdrant_client.search(
        collection_name="docs",
        query_vector=("content_vector", query_vector),
        limit=limit,
        with_payload=True
    )

    # Format results
    results = []
    for hit in search_result:
        results.append({
            "title": hit.payload.get("title", ""),
            "url": hit.payload.get("url", ""),
            "text": hit.payload.get("text", ""),
            "score": hit.score,
            "source": "knowledge_base"
        })

    return {
        "source": "knowledge_base",
        "results": results,
        "result_count": len(results)
    }
