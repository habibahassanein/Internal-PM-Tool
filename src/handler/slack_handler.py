from __future__ import annotations

import logging
import os
import time
import re
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# =========================
# Channel Intelligence (merged)
# =========================

@dataclass
class ChannelInfo:
    """Information about a Slack channel."""
    id: str
    name: str
    purpose: str
    topic: str
    is_private: bool
    is_member: bool
    member_count: int
    keywords: Set[str]
    patterns: Set[str]
    category: Optional[str] = None

@dataclass
class ChannelPattern:
    """Detected pattern in channel naming."""
    pattern: str  # e.g., "data-*", "team-*"
    regex: re.Pattern
    channels: Set[str]
    category: str
    confidence: float

class ChannelIntelligence:
    """
    Intelligent channel analysis and filtering for Slack workspaces.

    This class analyzes all accessible channels to:
    1. Discover channel patterns and categories
    2. Extract keywords from names, purposes, and topics
    3. Build an index for fast relevant channel lookup
    4. Cache results in class instance for performance
    """

    def __init__(self, client: WebClient):
        self.client = client
        self._channels: Dict[str, ChannelInfo] = {}
        self._patterns: List[ChannelPattern] = []
        self._keyword_index: Dict[str, Set[str]] = defaultdict(set)
        self._category_index: Dict[str, Set[str]] = defaultdict(set)
        self._initialized = False

    def initialize(self, force_refresh: bool = False) -> None:
        if self._initialized and not force_refresh:
            return
        logger.info("Initializing channel intelligence...")
        self._discover_channels()
        self._detect_patterns()
        self._build_keyword_index()
        self._initialized = True
        logger.info(f"Channel intelligence initialized: {len(self._channels)} channels, {len(self._patterns)} patterns")

    def _discover_channels(self) -> None:
        logger.info("Discovering accessible channels...")
        try:
            next_cursor: Optional[str] = None
            total_channels = 0

            while True:
                response = self.client.conversations_list(
                    types="public_channel,private_channel",
                    exclude_archived=True,
                    limit=1000,
                    cursor=next_cursor or None,
                )
                channels = response.get("channels", [])
                if not channels:
                    break

                for channel in channels:
                    channel_id = channel.get("id")
                    if not channel_id:
                        continue

                    # Skip DMs and group DMs
                    if channel.get("is_im") or channel.get("is_mpim"):
                        continue

                    # Skip excluded channels
                    channel_name = channel.get("name", "").lower()
                    if channel_name == "inhousezendesk":
                        continue

                    is_private = channel.get("is_private", False)
                    is_member = channel.get("is_member", False)
                    if is_private and not is_member:
                        continue

                    name = channel.get("name", "")
                    purpose = channel.get("purpose", {}).get("value", "")
                    topic = channel.get("topic", {}).get("value", "")
                    member_count = channel.get("num_members", 0)

                    keywords = self._extract_keywords_from_text(f"{name} {purpose} {topic}")

                    channel_info = ChannelInfo(
                        id=channel_id,
                        name=name,
                        purpose=purpose,
                        topic=topic,
                        is_private=is_private,
                        is_member=is_member,
                        member_count=member_count,
                        keywords=keywords,
                        patterns=set()
                    )
                    self._channels[channel_id] = channel_info
                    total_channels += 1

                next_cursor = (response.get("response_metadata") or {}).get("next_cursor") or ""
                if not next_cursor:
                    break

            logger.info(f"Discovered {total_channels} accessible channels")
        except SlackApiError as e:
            logger.error(f"Failed to discover channels: {e}")
            raise

    def _extract_keywords_from_text(self, text: str) -> Set[str]:
        if not text:
            return set()
        stopwords = {
            "the","a","an","and","or","but","in","on","at","to","for","of","with","by","from","is","are","was","were","be","been",
            "being","have","has","had","do","does","did","will","would","could","should","may","might","can","about","what","when",
            "where","why","how","who","which","this","that","these","those","channel","chat","discussion","team","group","project"
        }
        words = re.findall(r'\b[a-zA-Z0-9-]+\b', text.lower())
        keywords = set()
        for word in words:
            if word in stopwords or len(word) < 2:
                continue
            if word.isdigit():
                continue
            clean_word = word.strip('-')
            if len(clean_word) >= 2:
                keywords.add(clean_word)
        return keywords

    def _detect_patterns(self) -> None:
        logger.info("Detecting channel patterns...")
        prefix_groups = defaultdict(list)
        suffix_groups = defaultdict(list)

        for channel_id, channel_info in self._channels.items():
            name = channel_info.name.lower()

            # Prefix (e.g., "eng-", "data-")
            for i in range(2, len(name)):
                if name[i] == '-':
                    prefix = name[:i+1]
                    prefix_groups[prefix].append(channel_id)
                    break

            # Suffix (e.g., "-dev", "-prod")
            for i in range(len(name)-2, 0, -1):
                if name[i] == '-':
                    suffix = name[i:]
                    suffix_groups[suffix].append(channel_id)
                    break

        self._patterns = []

        for prefix, channel_ids in prefix_groups.items():
            if len(channel_ids) >= 2:
                category = self._infer_category_from_pattern(prefix, channel_ids)
                confidence = min(1.0, len(channel_ids) / 10.0)
                pattern = ChannelPattern(
                    pattern=prefix + "*",
                    regex=re.compile(f"^{re.escape(prefix[:-1])}-", re.IGNORECASE),
                    channels=set(channel_ids),
                    category=category,
                    confidence=confidence
                )
                self._patterns.append(pattern)
                for channel_id in channel_ids:
                    self._channels[channel_id].patterns.add(pattern.pattern)
                    self._channels[channel_id].category = category

        for suffix, channel_ids in suffix_groups.items():
            if len(channel_ids) >= 2:
                category = self._infer_category_from_pattern(suffix, channel_ids)
                confidence = min(1.0, len(channel_ids) / 10.0)
                pattern = ChannelPattern(
                    pattern="*" + suffix,
                    regex=re.compile(f"{re.escape(suffix[1:])}$", re.IGNORECASE),
                    channels=set(channel_ids),
                    category=category,
                    confidence=confidence
                )
                self._patterns.append(pattern)
                for channel_id in channel_ids:
                    self._channels[channel_id].patterns.add(pattern.pattern)
                    if not self._channels[channel_id].category:
                        self._channels[channel_id].category = category

        self._patterns.sort(key=lambda p: p.confidence, reverse=True)
        logger.info(f"Detected {len(self._patterns)} channel patterns")

    def _infer_category_from_pattern(self, pattern: str, channel_ids: List[str]) -> str:
        sample_names = [self._channels[cid].name.lower() for cid in channel_ids[:5]]
        sample_text = " ".join(sample_names)
        category_keywords = {
            "engineering": ["eng","dev","development","backend","frontend","api","code","tech"],
            "data": ["data","analytics","metrics","dashboard","viz","visualization","reporting"],
            "product": ["product","feature","roadmap","planning","strategy"],
            "design": ["design","ui","ux","mockup","prototype","wireframe"],
            "marketing": ["marketing","campaign","content","social","brand"],
            "sales": ["sales","revenue","customer","lead","prospect"],
            "support": ["support","help","customer","ticket","issue"],
            "operations": ["ops","operations","infrastructure","deployment","monitoring"],
            "security": ["security","auth","compliance","audit","privacy"],
            "qa": ["qa","test","testing","quality","validation"],
            "project": ["project","initiative","milestone","delivery"],
            "team": ["team","squad","group","department"],
            "release": ["release","announcement","announce","version","update","deploy","deployment","launch"]
        }
        best_category = "general"
        best_score = 0
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in sample_text)
            if score > best_score:
                best_score = score
                best_category = category
        return best_category

    def _build_keyword_index(self) -> None:
        logger.info("Building keyword index...")
        for channel_id, channel_info in self._channels.items():
            for keyword in channel_info.keywords:
                self._keyword_index[keyword].add(channel_id)
            if channel_info.category:
                self._category_index[channel_info.category].add(channel_id)
            name_words = self._extract_keywords_from_text(channel_info.name)
            for word in name_words:
                self._keyword_index[word].add(channel_id)
        logger.info(f"Built keyword index with {len(self._keyword_index)} keywords")

    def get_relevant_channels(
        self,
        query: str,
        top_k: int = 20,
        keywords: Optional[List[str]] = None,
        priority_terms: Optional[List[str]] = None
    ) -> List[str]:
        if not self._initialized:
            self.initialize()

        if not query and not keywords and not priority_terms:
            return self._get_most_active_channels(top_k)

        if not keywords:
            keywords = list(self._extract_keywords_from_text(query))
        if priority_terms:
            keywords.extend(priority_terms)

        channel_scores = defaultdict(float)

        for keyword in keywords:
            keyword_lower = keyword.lower()

            if keyword_lower in self._keyword_index:
                for channel_id in self._keyword_index[keyword_lower]:
                    channel_scores[channel_id] += 2.0

            for pattern in self._patterns:
                if pattern.regex.search(keyword_lower):
                    for channel_id in pattern.channels:
                        channel_scores[channel_id] += pattern.confidence * 1.0

            for indexed_keyword, channel_ids in self._keyword_index.items():
                if keyword_lower in indexed_keyword or indexed_keyword in keyword_lower:
                    for channel_id in channel_ids:
                        channel_scores[channel_id] += 0.5

            for channel_id, channel_info in self._channels.items():
                channel_name = channel_info.name.lower()
                if keyword_lower in channel_name:
                    channel_scores[channel_id] += 3.0
                elif any(word in channel_name for word in keyword_lower.split()):
                    channel_scores[channel_id] += 1.5

        if priority_terms:
            for term in priority_terms:
                term_lower = term.lower()
                if term_lower in self._keyword_index:
                    for channel_id in self._keyword_index[term_lower]:
                        channel_scores[channel_id] += 2.0

        sorted_channels = sorted(channel_scores.items(), key=lambda x: x[1], reverse=True)
        return [channel_id for channel_id, _ in sorted_channels[:top_k]]

    def _get_most_active_channels(self, top_k: int) -> List[str]:
        sorted_channels = sorted(self._channels.items(), key=lambda x: x[1].member_count, reverse=True)
        return [channel_id for channel_id, _ in sorted_channels[:top_k]]

    def get_channel_info(self, channel_id: str) -> Optional[ChannelInfo]:
        if not self._initialized:
            self.initialize()
        return self._channels.get(channel_id)

    def get_patterns(self) -> List[ChannelPattern]:
        if not self._initialized:
            self.initialize()
        return self._patterns

    def get_categories(self) -> Dict[str, int]:
        if not self._initialized:
            self.initialize()
        category_counts = defaultdict(int)
        for channel_info in self._channels.values():
            if channel_info.category:
                category_counts[channel_info.category] += 1
        return dict(category_counts)

# =========================
# Slack Search (merged)
# =========================

# Module-level caches
_slack_client: Optional[WebClient] = None
_user_cache: Dict[str, str] = {}
_channel_intelligence: Optional[ChannelIntelligence] = None

def _get_slack_client() -> WebClient:
    global _slack_client
    if _slack_client is None:
        token = os.getenv("SLACK_USER_TOKEN")
        if not token:
            raise RuntimeError("Missing SLACK_USER_TOKEN environment variable.")
        _slack_client = WebClient(token=token)
    return _slack_client

def get_channel_intelligence() -> ChannelIntelligence:
    global _channel_intelligence
    if _channel_intelligence is None:
        client = _get_slack_client()
        _channel_intelligence = ChannelIntelligence(client)
    _channel_intelligence.initialize()
    return _channel_intelligence

def _resolve_username(client: WebClient, user_id: Optional[str]) -> str:
    if not user_id:
        return "Unknown User"
    if user_id in _user_cache:
        return _user_cache[user_id]
    try:
        resp = client.users_info(user=user_id)
        if resp.get("ok"):
            profile = resp.get("user", {})
            username = (
                profile.get("real_name") or 
                profile.get("display_name") or 
                profile.get("name") or 
                profile.get("profile", {}).get("display_name") or
                profile.get("profile", {}).get("real_name") or
                f"User {user_id}"
            )
        else:
            username = f"User {user_id}"
        _user_cache[user_id] = username
        return username
    except Exception as e:
        logger.warning(f"Failed to resolve username for {user_id}: {e}")
        fallback = f"User {user_id}"
        _user_cache[user_id] = fallback
        return fallback

def _format_slack_timestamp(ts: str) -> str:
    try:
        if not ts:
            return "Unknown date"
        timestamp = float(ts)
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return "Unknown date"

def strategy_1_native_slack_search(user_query: str) -> List[Dict[str, Any]]:
    logger.info(f"Strategy 1: Native Slack search for: {user_query}")
    client = _get_slack_client()
    results = []
    try:
        response = client.search_messages(
            query=user_query,
            count=1000,
            sort="score"
        )
        messages = response.get("messages", {})
        matches = messages.get("matches", [])
        logger.info(f"Native Slack search returned {len(matches)} matches")
        for match in matches:
            text = match.get("text", "")
            if not text:
                continue
            channel_info = match.get("channel", {})
            channel_name = channel_info.get("name", "unknown")
            channel_id = channel_info.get("id", "")
            if channel_info.get("is_im") or channel_info.get("is_mpim"):
                continue
            if channel_name.lower() == "inhousezendesk":
                continue
            user_id = match.get("user")
            username = _resolve_username(client, user_id)
            ts = match.get("ts")
            permalink = match.get("permalink", "")
            results.append({
                "text": text,
                "username": username,
                "channel": channel_name,
                "channel_id": channel_id,
                "ts": ts,
                "date": _format_slack_timestamp(ts),
                "permalink": permalink,
                "strategy": "native_slack",
                "slack_score": match.get("score", 0.0)
            })
        logger.info(f"Strategy 1: Returning {len(results)} results (no filtering applied)")
        return results
    except SlackApiError as e:
        logger.error(f"Strategy 1: Slack API error: {e}")
        return []
    except Exception as e:
        logger.error(f"Strategy 1: Unexpected error: {e}")
        return []

def strategy_2_smart_targeted_search(
    user_query: str,
    intent_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    logger.info(f"Strategy 2: Smart targeted search for: {user_query}")
    client = _get_slack_client()
    intelligence = get_channel_intelligence()

    slack_params = intent_data.get("slack_params", {})
    keywords = slack_params.get("keywords", [])
    priority_terms = slack_params.get("priority_terms", [])

    relevant_channel_ids = intelligence.get_relevant_channels(
        query=user_query,
        top_k=10,
        keywords=keywords,
        priority_terms=priority_terms
    )
    if not relevant_channel_ids:
        logger.info("Strategy 2: No relevant channels found")
        return []

    logger.info(f"Strategy 2: Searching {len(relevant_channel_ids)} relevant channels")
    all_results = []
    max_workers = min(5, len(relevant_channel_ids))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_channel = {
            executor.submit(
                _search_channel_all_history,
                client,
                channel_id,
                user_query,
                keywords,
                priority_terms
            ): channel_id
            for channel_id in relevant_channel_ids
        }
        try:
            for future in as_completed(future_to_channel, timeout=30):
                channel_id = future_to_channel[future]
                try:
                    channel_results = future.result(timeout=10)
                    all_results.extend(channel_results)
                except Exception as e:
                    logger.warning(f"Strategy 2: Failed to search channel {channel_id}: {e}")
        except TimeoutError:
            logger.warning("Strategy 2: Timeout waiting for channel search results")
            for future in future_to_channel:
                future.cancel()

    logger.info(f"Strategy 2: Found {len(all_results)} results from targeted channels")
    return all_results

def _search_channel_all_history(
    client: WebClient,
    channel_id: str,
    user_query: str,
    keywords: List[str],
    priority_terms: List[str]
) -> List[Dict[str, Any]]:
    results = []
    try:
        channel_info = client.conversations_info(channel=channel_id)
        channel_data = channel_info.get("channel", {})
        channel_name = channel_data.get("name", f"Channel {channel_id}")

        if channel_data.get("is_im") or channel_data.get("is_mpim"):
            return []
        if channel_name.lower() == "inhousezendesk":
            return []

        next_cursor: Optional[str] = None
        total_fetched = 0
        max_messages = 100

        while total_fetched < max_messages:
            response = client.conversations_history(
                channel=channel_id,
                limit=min(100, max_messages - total_fetched),
                cursor=next_cursor or None,
                inclusive=True
            )
            messages = response.get("messages", [])
            if not messages:
                break

            total_fetched += len(messages)

            for msg in messages:
                text = msg.get("text", "")
                if not text:
                    continue
                if msg.get("subtype") in ["channel_join","channel_leave","channel_topic","channel_purpose"]:
                    continue

                relevance_score = _calculate_semantic_relevance(text, user_query, keywords, priority_terms)
                if relevance_score > 0:
                    user_id = msg.get("user")
                    username = _resolve_username(client, user_id)
                    ts = msg.get("ts")
                    try:
                        permalink_resp = client.chat_getPermalink(channel=channel_id, message_ts=ts)
                        permalink = permalink_resp.get("permalink", "")
                    except Exception:
                        permalink = ""

                    results.append({
                        "text": text,
                        "username": username,
                        "channel": channel_name,
                        "channel_id": channel_id,
                        "ts": ts,
                        "date": _format_slack_timestamp(ts),
                        "permalink": permalink,
                        "semantic_score": relevance_score,
                        "strategy": "smart_targeted"
                    })

            next_cursor = (response.get("response_metadata") or {}).get("next_cursor") or ""
            if not next_cursor:
                break

        logger.info(f"Channel {channel_name}: Found {len(results)} relevant messages from {total_fetched} total")
    except SlackApiError as e:
        logger.warning(f"Failed to search channel {channel_id}: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error searching channel {channel_id}: {e}")

    return results

def _calculate_semantic_relevance(
    text: str,
    user_query: str,
    keywords: List[str],
    priority_terms: List[str]
) -> float:
    if not text:
        return 0.0

    text_lower = text.lower()
    query_lower = user_query.lower()
    score = 0.0

    if query_lower in text_lower:
        score += 20.0

    for term in priority_terms:
        term_lower = term.lower()
        if term_lower in text_lower:
            if re.search(r'\b' + re.escape(term_lower) + r'\b', text_lower):
                score += 15.0
            else:
                score += 10.0

    for keyword in keywords:
        keyword_lower = keyword.lower()
        if keyword_lower in text_lower:
            if re.search(r'\b' + re.escape(keyword_lower) + r'\b', text_lower):
                score += 8.0
            else:
                score += 5.0

    query_terms = [term for term in query_lower.split() if len(term) > 2]
    matched_terms = 0
    for term in query_terms:
        if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
            score += 2.0
            matched_terms += 1
    if matched_terms > 0:
        coverage_bonus = (matched_terms / len(query_terms)) * 10.0
        score += coverage_bonus

    semantic_patterns = _generate_semantic_patterns(query_lower)
    for pattern in semantic_patterns:
        if re.search(pattern, text_lower):
            score += 5.0

    technical_entities = _extract_technical_entities(query_lower)
    text_entities = _extract_technical_entities(text_lower)
    for entity in technical_entities:
        if entity in text_entities:
            score += 12.0
        else:
            similarity = _calculate_entity_similarity(entity, text_entities)
            if similarity > 0.7:
                score += similarity * 8.0

    context_score = _calculate_contextual_relevance(text_lower, query_lower)
    score += context_score

    return score

def _generate_semantic_patterns(query: str) -> List[str]:
    patterns = []
    if 'release' in query:
        patterns.extend([
            r'\brelease\b.*\b(?:date|schedule|announcement|available|ga)\b',
            r'\b(?:date|schedule|announcement|available|ga)\b.*\brelease\b',
            r'\bgo\s+ga\b',
            r'\bgeneral\s+availability\b'
        ])
    if 'delivery' in query:
        patterns.extend([
            r'\bdelivery\b.*\b(?:date|schedule|timeline)\b',
            r'\b(?:date|schedule|timeline)\b.*\bdelivery\b'
        ])
    if any(word in query for word in ['version','v','update']):
        patterns.extend([
            r'\bversion\b.*\d+\.\d+',
            r'\bv\d+\.\d+',
            r'\bupdate\b.*\b(?:available|released)\b'
        ])
    if 'date' in query:
        patterns.extend([
            r'\b(?:date|schedule|timeline)\b.*\b(?:release|delivery|available)\b',
            r'\b(?:release|delivery|available)\b.*\b(?:date|schedule|timeline)\b'
        ])
    return patterns

def _extract_technical_entities(text: str) -> List[str]:
    entities = []
    version_patterns = [
        r'\b\d{4}\.\d+\.\d+\b',
        r'\b\d+\.\d+\.\d+\b',
        r'\bv\d+\.\d+(?:\.\d+)?\b',
        r'\bversion\s+\d+\.\d+(?:\.\d+)?\b'
    ]
    for pattern in version_patterns:
        entities.extend(re.findall(pattern, text, re.IGNORECASE))
    code_patterns = [
        r'\b[A-Z]+-\d+\b',
        r'\b[A-Z]{2,}\d+\b'
    ]
    for pattern in code_patterns:
        entities.extend(re.findall(pattern, text, re.IGNORECASE))
    return entities

def _calculate_entity_similarity(entity1: str, entities2: List[str]) -> float:
    if not entities2:
        return 0.0
    def extract_version_parts(version_str):
        parts = re.findall(r'\d+', version_str)
        return [int(p) for p in parts] if parts else []
    parts1 = extract_version_parts(entity1)
    if not parts1:
        return 0.0
    max_similarity = 0.0
    for entity2 in entities2:
        parts2 = extract_version_parts(entity2)
        if not parts2:
            continue
        if len(parts1) >= 2 and len(parts2) >= 2:
            if parts1[0] == parts2[0] and parts1[1] == parts2[1]:
                similarity = 0.9
            elif len(parts1) >= 3 and len(parts2) >= 3 and parts1[1] == parts2[1] and parts1[2] == parts2[2]:
                similarity = 0.8
            elif parts1[1] == parts2[1]:
                similarity = 0.6
            elif len(parts1) >= 3 and len(parts2) >= 3 and parts1[2] == parts2[2]:
                similarity = 0.4
            else:
                similarity = 0.0
        else:
            similarity = 0.0
        max_similarity = max(max_similarity, similarity)
    return max_similarity

def _calculate_contextual_relevance(text: str, query: str) -> float:
    score = 0.0
    if '?' in text and any(word in query for word in ['what','when','where','how','why']):
        score += 3.0
    if any(word in query for word in ['date','when','schedule','timeline']):
        if any(word in text for word in ['date','schedule','timeline','when','available']):
            score += 4.0
    if any(word in query for word in ['status','available','released','ga']):
        if any(word in text for word in ['available','released','ga','status','ready']):
            score += 4.0
    return score

def _analyze_time_intent(user_query: str) -> Dict[str, Any]:
    query_lower = user_query.lower()
    high_temporal_patterns = [
        r'\b(?:version|v)\s*\d{4}\.\d+\.\d+\b',
        r'\b(?:latest|current|newest|most recent)\b',
        r'\b(?:today|yesterday|this week|this month)\b',
        r'\b(?:just|recently|recent)\b'
    ]
    for pattern in high_temporal_patterns:
        if re.search(pattern, query_lower):
            return {"temporal_specificity": "high", "boost_factor": 5.0}
    medium_temporal_patterns = [
        r'\b(?:when|date|schedule|timeline)\b',
        r'\b(?:release|announcement|update)\b'
    ]
    for pattern in medium_temporal_patterns:
        if re.search(pattern, query_lower):
            return {"temporal_specificity": "medium", "boost_factor": 2.0}
    return {"temporal_specificity": "low", "boost_factor": 0.5}

def _calculate_adaptive_recency_boost(
    ts: str,
    user_query: str,
    intent_data: Dict[str, Any]
) -> float:
    try:
        timestamp = float(ts)
        days_ago = (time.time() - timestamp) / (24 * 3600)
    except (ValueError, TypeError):
        return 1.0
    time_intent = _analyze_time_intent(user_query)
    boost_factor = time_intent["boost_factor"]

    if time_intent["temporal_specificity"] == "high":
        if days_ago < 1:
            recency_decay = 1.0
        elif days_ago < 7:
            recency_decay = 0.8
        elif days_ago < 30:
            recency_decay = 0.6
        elif days_ago < 90:
            recency_decay = 0.4
        else:
            recency_decay = 0.2
    elif time_intent["temporal_specificity"] == "medium":
        if days_ago < 7:
            recency_decay = 1.0
        elif days_ago < 30:
            recency_decay = 0.8
        elif days_ago < 90:
            recency_decay = 0.6
        else:
            recency_decay = 0.4
    else:
        if days_ago < 30:
            recency_decay = 1.0
        elif days_ago < 90:
            recency_decay = 0.9
        else:
            recency_decay = 0.8

    adaptive_boost = boost_factor * recency_decay
    return adaptive_boost

def _deduplicate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_hashes: Set[int] = set()
    unique_results = []
    for result in results:
        ts = result.get("ts", "")
        channel_id = result.get("channel_id", "")
        result_hash = hash(f"{ts}_{channel_id}")
        if result_hash not in seen_hashes:
            seen_hashes.add(result_hash)
            unique_results.append(result)
    return unique_results

def _calculate_final_score(
    result: Dict[str, Any],
    user_query: str,
    intent_data: Dict[str, Any]
) -> float:
    semantic_score = result.get("semantic_score", 0.0)
    slack_score = result.get("slack_score", 0.0)

    if semantic_score == 0.0:
        slack_params = intent_data.get("slack_params", {})
        keywords = slack_params.get("keywords", [])
        priority_terms = slack_params.get("priority_terms", [])
        semantic_score = _calculate_semantic_relevance(
            result.get("text", ""),
            user_query,
            keywords,
            priority_terms
        )
    adaptive_boost = _calculate_adaptive_recency_boost(
        result.get("ts", "0"),
        user_query,
        intent_data
    )
    alpha = 0.7
    beta = 0.3
    base_score = (alpha * semantic_score) + (beta * slack_score)
    final_score = base_score * adaptive_boost
    return final_score

def _rank_results(
    results: List[Dict[str, Any]],
    user_query: str,
    intent_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    logger.info(f"Ranking {len(results)} results using adaptive ranking algorithm")
    scored_results = []
    for result in results:
        final_score = _calculate_final_score(result, user_query, intent_data)
        scored_results.append((final_score, result))
    scored_results.sort(key=lambda x: x[0], reverse=True)
    final_results = []
    for _, result in scored_results:
        result.pop("semantic_score", None)
        result.pop("slack_score", None)
        final_results.append(result)
    logger.info(f"Ranked {len(final_results)} results")
    return final_results

def search_slack_simplified(
    user_query: str,
    intent_data: Dict[str, Any],
    max_total_results: int = 15
) -> List[Dict[str, Any]]:
    logger.info(f"Starting max-retrieval, adaptive-ranking search for: {user_query}")
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_strategy_1 = executor.submit(strategy_1_native_slack_search, user_query)
        future_strategy_2 = executor.submit(strategy_2_smart_targeted_search, user_query, intent_data)

        strategy_1_results = []
        strategy_2_results = []

        try:
            strategy_1_results = future_strategy_1.result(timeout=30)
            logger.info(f"Strategy 1: Retrieved {len(strategy_1_results)} results")
        except Exception as e:
            logger.error(f"Strategy 1 failed: {e}")

        try:
            strategy_2_results = future_strategy_2.result(timeout=30)
            logger.info(f"Strategy 2: Retrieved {len(strategy_2_results)} results")
        except Exception as e:
            logger.error(f"Strategy 2 failed: {e}")

    all_results = strategy_1_results + strategy_2_results
    logger.info(f"Combined: {len(all_results)} total results")
    if not all_results:
        logger.warning("No results from either strategy")
        return []

    unique_results = _deduplicate_results(all_results)
    logger.info(f"After deduplication: {len(unique_results)} unique results")

    ranked_results = _rank_results(unique_results, user_query, intent_data)

    if len(ranked_results) > max_total_results:
        logger.info(f"Limiting to TOP {max_total_results} messages from {len(ranked_results)} total")
        ranked_results = ranked_results[:max_total_results]

    logger.info(f"Final results: {len(ranked_results)} messages sent to UI")
    return ranked_results

# Legacy compatibility function - used by app.py and agent tools

def search_slack_messages(
    query: str,
    max_results: int = 10,
    channel_filter: Optional[str] = None,
    max_age_hours: int = 168
) -> List[dict]:
    if not query or not query.strip():
        return []
    query_terms = query.lower().split()
    keywords = [term for term in query_terms if len(term) > 2]
    priority_terms = keywords[:3]

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

    results = search_slack_simplified(query, intent_data, max_results)

    legacy_results = []
    for result in results:
        legacy_results.append({
            "text": result.get("text", ""),
            "username": result.get("username", "Unknown"),
            "channel": result.get("channel", "unknown"),
            "ts": result.get("ts", ""),
            "permalink": result.get("permalink", ""),
            "source": "slack"
        })
    return legacy_results

__all__ = [
    "ChannelIntelligence",
    "ChannelInfo",
    "ChannelPattern",
    "get_channel_intelligence",
    "search_slack_messages",
    "search_slack_simplified",
    "strategy_1_native_slack_search",
    "strategy_2_smart_targeted_search"
]