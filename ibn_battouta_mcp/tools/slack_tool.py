"""
Slack search tool for MCP server.
Wraps existing slack_handler functionality.
"""
import sys
from typing import Dict, Any, List
from pathlib import Path

parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from src.handler.slack_handler import search_slack_simplified
from ibn_battouta_mcp.context.user_context import user_context


def search_slack(arguments: Dict[str, Any]) -> dict:
    """
    Search Slack messages with intent-based matching.

    Args:
        query (str): Search query
        max_results (int): Maximum number of results (default: 10)
        channel_filter (str, optional): Specific channel to search
        max_age_hours (int): Maximum age of messages in hours (default: 168 = 1 week)

    Returns:
        dict: Search results with message metadata
    """
    query = arguments["query"]
    max_results = arguments.get("max_results", 10)
    channel_filter = arguments.get("channel_filter")
    max_age_hours = arguments.get("max_age_hours", 168)

    # Build intent data (from your main.py logic)
    query_terms = query.lower().split()
    keywords = [term for term in query_terms if len(term) > 2]
    priority_terms = keywords[:3]  # Top 3 terms

    intent_data = {
        "slack_params": {
            "keywords": keywords,
            "priority_terms": priority_terms,
            "channels": channel_filter if channel_filter else "all",
            "time_range": "all",
            "limit": max_results
        },
        "search_strategy": "fuzzy_match"
    }

    token = user_context.get().get("slack_token")
    if not token:
        raise RuntimeError(
            "Slack search requires a user token. "
            "Provide it via the 'slack-token' header when calling the MCP server."
        )
    # Call your existing handler with explicit token
    results = search_slack_simplified(query, intent_data, max_results, user_token=token)

    # Format results
    formatted_results = []
    for result in results:
        formatted_results.append({
            "text": result.get("text", ""),
            "username": result.get("username", "Unknown"),
            "channel": result.get("channel", "unknown"),
            "ts": result.get("ts", ""),
            "permalink": result.get("permalink", ""),
            "source": "slack"
        })

    return {
        "source": "slack",
        "results": formatted_results,
        "result_count": len(formatted_results)
    }
