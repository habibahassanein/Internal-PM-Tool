
import sys
from typing import Dict, Any, List
from pathlib import Path

from handlers.slack_handler import search_slack_simplified
from context.user_context import user_context
from auth import context
from auth.session_store import get_session_store

import logging
logger = logging.getLogger(__name__)


def get_slack_token() -> str:
    """
    Get the Slack token for the current session.
    
    First checks user_context (for OAuth callback flow),
    then falls back to session store lookup.
    
    Returns:
        Slack access token
        
    Raises:
        RuntimeError: If no valid token found
    """
    # First, check user_context (set during OAuth callback)
    token = user_context.get().get("slack_token")
    if token:
        logger.info("Got token from user_context")
        return token
    
    # Fall back to session store lookup
    session_id = context.fastmcp_session_id.get()
    user_id = context.authenticated_user_id.get()
    
    logger.info(f"Session lookup: session_id={session_id}, user_id={user_id}")
    
    if not session_id:
        raise RuntimeError(
            "No session found. Please authenticate first using the slack_get_oauth_url tool."
        )
    
    if not user_id:
        raise RuntimeError(
            "Session not authenticated. Please complete the OAuth flow using slack_get_oauth_url."
        )
    
    store = get_session_store()
    token = store.get_user_token_with_validation(user_id, session_id)
    
    if not token:
        raise RuntimeError(
            "No valid Slack token found for this session. Please re-authenticate using slack_get_oauth_url."
        )
    
    logger.info(f"Got token from session store for user {user_id}")
    return token


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

    # Get token from session store (handles OAuth flow properly)
    token = get_slack_token()
    
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
