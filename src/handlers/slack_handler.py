
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

logger = logging.getLogger(__name__)

_CLIENT_CACHE: Dict[str, WebClient] = {}


def _get_slack_client(user_token: Optional[str] = None) -> WebClient:
    """Get Slack client; requires authenticated user's token."""
    token = user_token
    if not token:
        raise RuntimeError("Missing Slack token. Provide it to the MCP server.")

    cache_key = f"slack_client_{hash(token)}"
    client = _CLIENT_CACHE.get(cache_key)
    if client is None:
        client = WebClient(token=token)
        _CLIENT_CACHE[cache_key] = client
    return client


def search_slack_simplified(
    query: str,
    intent_data: Optional[Dict] = None,
    max_results: int = 15,
    user_token: Optional[str] = None
) -> List[Dict]:
    """
    Simplified Slack search for MCP server.
    Uses native Slack search API without channel intelligence.
    """
    logger.info(f"Searching Slack for: {query}")

    client = _get_slack_client(user_token)
    results = []

    try:
        response = client.search_messages(
            query=query,
            count=max_results,
            sort="score"
        )

        messages = response.get("messages", {})
        matches = messages.get("matches", [])

        logger.info(f"Found {len(matches)} Slack messages")

        for match in matches[:max_results]:
            text = match.get("text", "")
            if not text:
                continue

            channel_info = match.get("channel", {})
            channel_name = channel_info.get("name", "unknown")

            # Skip private DMs
            if channel_info.get("is_im") or channel_info.get("is_mpim"):
                continue

            results.append({
                "text": text,
                "username": match.get("username", "Unknown"),
                "channel": channel_name,
                "ts": match.get("ts", ""),
                "permalink": match.get("permalink", ""),
                "score": match.get("score", 0.0)
            })

        logger.info(f"Returning {len(results)} Slack results")

    except SlackApiError as e:
        logger.error(f"Slack API error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

    return results


__all__ = ["search_slack_simplified"]
