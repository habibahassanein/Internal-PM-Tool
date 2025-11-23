"""
Confluence search tool for MCP server.
Wraps existing confluence_handler functionality.
"""
import sys
import os
from typing import Dict, Any
from pathlib import Path

# Add parent directory to path for src imports
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from src.handler.confluence_handler import search_confluence_pages
from ibn_battouta_mcp.context.user_context import user_context


def search_confluence(arguments: Dict[str, Any]) -> dict:
    """
    Search Confluence pages with optimized query processing.

    Args:
        query (str): Search query
        max_results (int): Maximum number of results (default: 10)
        space_filter (str, optional): Specific Confluence space to search

    Returns:
        dict: Search results with page metadata
    """
    query = arguments["query"]
    max_results = arguments.get("max_results", 10)
    space_filter = arguments.get("space_filter")

    # Context contains Confluence credentials if needed
    # Your existing handler uses environment variables, so this should work

    # Apply stop word filtering (from your main.py logic)
    query_words = set(query.lower().split())
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
        "with", "by", "is", "are", "was", "were", "be", "been", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should", "may",
        "might", "can", "what", "when", "where", "why", "how", "who", "which",
        "updates", "about", "on"
    }
    distinct_words = [word for word in query_words if word not in stop_words and len(word) > 2]

    # Build optimized search query
    search_query = " ".join(distinct_words) if distinct_words else query

    # Call your existing handler
    results = search_confluence_pages(search_query, max_results, space_filter)

    return {
        "source": "confluence",
        "results": results,
        "query_optimized": search_query,
        "result_count": len(results)
    }
