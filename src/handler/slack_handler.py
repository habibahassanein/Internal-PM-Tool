"""
Slack integration handler for the Internal PM Tool.
Provides search functionality across Slack channels and messages.
"""

import logging
import time
import re
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, wait
import threading

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import streamlit as st

logger = logging.getLogger(__name__)

# Thread-safe caches
_user_cache_lock = threading.Lock()
_channel_cache_lock = threading.Lock()
_user_cache: Dict[str, str] = {}
_channel_info_cache: Dict[str, Dict] = {}


def _get_slack_client() -> WebClient:
    """Get Slack client with token from Streamlit secrets."""
    token = st.secrets.get("SLACK_USER_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing SLACK_USER_TOKEN in Streamlit secrets. "
            "Please add your Slack user token (xoxp-...) to Streamlit secrets."
        )
    return WebClient(token=token)


def _resolve_username(client: WebClient, user_id: Optional[str]) -> str:
    """Thread-safe username resolution with caching."""
    if not user_id:
        return "Unknown"
    
    with _user_cache_lock:
        if user_id in _user_cache:
            return _user_cache[user_id]
    
    try:
        resp = client.users_info(user=user_id)
        profile = resp.get("user", {})
        username = profile.get("real_name") or profile.get("name") or user_id
        
        with _user_cache_lock:
            _user_cache[user_id] = username
        
        return username
    except SlackApiError as e:
        logger.warning("Failed to resolve username for %s: %s", user_id, e)
        return user_id


def _get_channel_info(client: WebClient, channel_id: str) -> Dict:
    """Thread-safe channel info retrieval with caching."""
    with _channel_cache_lock:
        if channel_id in _channel_info_cache:
            return _channel_info_cache[channel_id]
    
    try:
        info = client.conversations_info(channel=channel_id)
        channel_data = info.get("channel", {})
        
        with _channel_cache_lock:
            _channel_info_cache[channel_id] = channel_data
        
        return channel_data
    except Exception as e:
        logger.warning(f"Failed to get info for channel {channel_id}: {e}")
        return {"id": channel_id, "name": channel_id}


def _get_channel_id(client: WebClient, channel_name_or_id: str) -> Optional[str]:
    """Resolve channel name to ID with caching."""
    if not channel_name_or_id:
        return None

    # Already an ID
    if channel_name_or_id.startswith(("C", "G")) and " " not in channel_name_or_id:
        return channel_name_or_id

    channel_name = channel_name_or_id.lstrip('#')
    
    # Skip placeholder values
    if channel_name.lower() in {"specific_channel_name", "channel_name", "none", ""}:
        return None

    logger.info(f"Looking for channel: '{channel_name}'")
    
    try:
        next_cursor: Optional[str] = None
        
        while True:
            response = client.conversations_list(
                types="public_channel,private_channel",
                limit=1000,
                cursor=next_cursor or None,
            )
            channels = response.get("channels", [])
            
            for channel in channels:
                ch_name = channel.get("name", "")
                ch_id = channel.get("id", "")
                
                if ch_name == channel_name:
                    logger.info(f"Found channel '{channel_name}' with ID: {ch_id}")
                    return ch_id

            next_cursor = (response.get("response_metadata") or {}).get("next_cursor") or ""
            if not next_cursor:
                break

        logger.warning(f"Channel '{channel_name}' not found")
        return None

    except SlackApiError as e:
        logger.error(f"Failed to resolve channel name '{channel_name}': {e}")
        return None


def _get_all_channels(limit: int = 200) -> List[str]:
    """Get all accessible channels (cached per session)."""
    # Use Streamlit session state for caching
    if "slack_channel_list" not in st.session_state:
        client = _get_slack_client()
        channel_ids = []
        
        try:
            next_cursor: Optional[str] = None
            while len(channel_ids) < limit:
                response = client.conversations_list(
                    types="public_channel,private_channel",
                    exclude_archived=True,
                    limit=200,
                    cursor=next_cursor or None,
                )
                channels = response.get("channels", [])
                
                for channel in channels:
                    channel_id = channel.get("id")
                    is_member = channel.get("is_member", False)
                    is_private = channel.get("is_private", False)
                    is_im = channel.get("is_im", False)
                    is_mpim = channel.get("is_mpim", False)
                    
                    if not is_im and not is_mpim and (not is_private or is_member):
                        channel_ids.append(channel_id)
                        if len(channel_ids) >= limit:
                            break
                
                next_cursor = (response.get("response_metadata") or {}).get("next_cursor") or ""
                if not next_cursor:
                    break
            
            st.session_state["slack_channel_list"] = channel_ids[:limit]
            logger.info(f"Found {len(st.session_state['slack_channel_list'])} accessible channels")
        except SlackApiError as e:
            logger.error(f"Failed to get channel list: {e}")
            st.session_state["slack_channel_list"] = []
    
    return st.session_state["slack_channel_list"]


def calculate_message_relevance(
    text: str,
    keywords: List[str],
    priority_terms: List[str],
    search_strategy: str = "fuzzy_match"
) -> float:
    """Calculate relevance score for a message (0-100)."""
    if not text:
        return 0.0
    
    text_lower = text.lower()
    score = 0.0
    
    # Priority terms scoring
    if priority_terms:
        priority_matches = sum(1 for term in priority_terms if term.lower() in text_lower)
        if search_strategy == "exact_match" and priority_matches == 0:
            return 0.0
        score += priority_matches * 20
    
    # Keyword matching
    if keywords:
        keyword_matches = sum(1 for kw in keywords if kw.lower() in text_lower)
        score += keyword_matches * 5
        
        if keyword_matches > 0:
            keyword_density = keyword_matches / len(keywords)
            score += keyword_density * 10
    
    # Exact phrase matching bonus
    for kw in keywords + priority_terms:
        if len(kw) > 3 and re.search(r'\b' + re.escape(kw.lower()) + r'\b', text_lower):
            score += 3
    
    # Technical patterns
    technical_patterns = [
        r'\bv?\d+\.\d+(?:\.\d+)?(?:\.\d+)?\b',
        r'\b[A-Z]+-\d+\b',
        r'\b[A-Z]{2,}\d+\b',
    ]
    for pattern in technical_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            score += 5
    
    return min(score, 100.0)


def search_slack_messages(
    query: str,
    max_results: int = 10,
    channel_filter: Optional[str] = None,
    max_age_hours: int = 168  # 1 week default
) -> List[dict]:
    """
    Search Slack messages with enhanced relevance scoring.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        channel_filter: Specific channel to search (None for all channels)
        max_age_hours: Maximum age of messages in hours
    
    Returns:
        List of message dictionaries with metadata
    """
    if not query or not query.strip():
        return []

    client = _get_slack_client()
    results = []
    
    try:
        # Determine channels to search
        channel_ids: List[str] = []
        
        if channel_filter and channel_filter.strip():
            channel_id = _get_channel_id(client, channel_filter.strip())
            if channel_id:
                channel_ids = [channel_id]
        else:
            logger.info("Searching all accessible channels")
            channel_ids = _get_all_channels(limit=200)

        if not channel_ids:
            logger.warning("No channels to search")
            return []

        # Calculate time window
        now_ts = time.time()
        oldest = now_ts - (max_age_hours * 3600) if max_age_hours > 0 else 0

        # Extract keywords from query
        query_terms = query.lower().split()
        keywords = [term for term in query_terms if len(term) > 2]
        priority_terms = keywords[:3]  # Top 3 terms as priority

        # Search across channels
        all_messages: List[Dict] = []
        
        # Use thread pool for parallel channel searching
        max_workers = min(10, len(channel_ids))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_channel = {}
            
            for channel_id in channel_ids:
                future = executor.submit(
                    _search_channel_messages,
                    client,
                    channel_id,
                    oldest,
                    keywords,
                    priority_terms,
                    max_results * 2  # Get more to filter later
                )
                future_to_channel[future] = channel_id
            
            # Wait for results with timeout
            done, not_done = wait(future_to_channel.keys(), timeout=20)
            
            # Cancel any remaining futures
            for future in not_done:
                future.cancel()
            
            # Collect results from completed futures
            for future in done:
                try:
                    channel_results = future.result(timeout=1)
                    all_messages.extend(channel_results)
                except Exception as e:
                    logger.warning(f"Channel search failed: {e}")

        logger.info(f"Total messages found: {len(all_messages)}")

        if not all_messages:
            return []

        # Sort by relevance then timestamp
        all_messages.sort(key=lambda x: (-x.get("relevance", 0), -x.get("_timestamp", 0)))

        # Enrich and format results
        for item in all_messages[:max_results]:
            channel_id = item["channel_id"]
            channel_info = _get_channel_info(client, channel_id)
            channel_name = channel_info.get("name", channel_id)
            username = _resolve_username(client, item.get("user_id"))
            
            try:
                permalink_resp = client.chat_getPermalink(channel=channel_id, message_ts=item.get("ts"))
                permalink = permalink_resp.get("permalink", "")
            except Exception:
                permalink = ""

            results.append({
                "text": item["text"],
                "username": username,
                "channel": channel_name,
                "ts": item["ts"],
                "permalink": permalink,
                "source": "slack"
            })

        logger.info(f"Returning {len(results)} relevant Slack messages")
        return results

    except Exception as e:
        logger.error(f"Slack search failed: {e}")
        return []


def _search_channel_messages(
    client: WebClient,
    channel_id: str,
    oldest: float,
    keywords: List[str],
    priority_terms: List[str],
    max_messages: int = 100
) -> List[Dict]:
    """Search a single channel for relevant messages."""
    results = []
    
    try:
        next_cursor: Optional[str] = None
        total_fetched = 0
        
        while total_fetched < max_messages:
            response = client.conversations_history(
                channel=channel_id,
                limit=min(100, max_messages - total_fetched),
                cursor=next_cursor or None,
                inclusive=True,
                oldest=str(oldest) if oldest > 0 else None
            )
            
            messages = response.get("messages", [])
            if not messages:
                break
            
            total_fetched += len(messages)
            
            for msg in messages:
                text = msg.get("text", "")
                if not text:
                    continue
                
                # Skip system messages
                if msg.get("subtype") in ["channel_join", "channel_leave", "channel_topic", "channel_purpose"]:
                    continue
                
                # Calculate relevance
                relevance_score = calculate_message_relevance(
                    text, keywords, priority_terms, "fuzzy_match"
                )
                
                if relevance_score > 10:  # Minimum relevance threshold
                    try:
                        ts_val = float(msg.get("ts", "0"))
                    except Exception:
                        ts_val = 0.0
                    
                    results.append({
                        "text": text,
                        "user_id": msg.get("user"),
                        "channel_id": channel_id,
                        "ts": msg.get("ts"),
                        "relevance": relevance_score,
                        "_timestamp": ts_val,
                    })
            
            next_cursor = (response.get("response_metadata") or {}).get("next_cursor") or ""
            if not next_cursor:
                break
        
        logger.info(f"Channel {channel_id}: Found {len(results)} relevant messages from {total_fetched} fetched")
        
    except SlackApiError as e:
        logger.warning(f"Failed to fetch from channel {channel_id}: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error in channel {channel_id}: {e}")
    
    return results


def search_slack_simple(query: str, max_results: int = 10) -> List[dict]:
    """Simple Slack search using search API."""
    if not query or not query.strip():
        return []

    client = _get_slack_client()
    results = []
    
    try:
        resp = client.search_messages(query=query, count=max_results)
        messages = resp.get("messages", {})
        matches = messages.get("matches", [])
        
        for item in matches:
            text = item.get("text", "")
            channel_name = (item.get("channel", {}) or {}).get("name", "unknown")
            ts = item.get("ts") or (item.get("message", {}) or {}).get("ts")
            permalink = item.get("permalink")
            username = item.get("username") or _resolve_username(client, item.get("user"))

            results.append({
                "text": text,
                "username": username,
                "channel": channel_name,
                "ts": ts,
                "permalink": permalink,
                "source": "slack"
            })

    except SlackApiError as e:
        logger.error("Slack API error during search: %s", e)
    except Exception as e:
        logger.exception("Unexpected error during Slack search: %s", e)

    return results[:max_results]
