import os
import html
import datetime

import streamlit as st
from dotenv import load_dotenv

from src.storage.cache_manager import (
    get_cached_search_results,
    cache_search_results,
    get_cache_manager
)
from src.agent import create_pm_agent

# =========================
# Environment & Config
# =========================

load_dotenv()

# Page config (do this as early as possible)
st.set_page_config(
    page_title="Internal PM Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants / Tunables
LLM_NAME = "Gemini 2.0 Flash Experimental"

# =========================
# Secrets & Keys
# =========================

def _get_secret_or_env(name: str, default: str = "") -> str:
    if name in st.secrets:
        return st.secrets.get(name, default)
    return os.getenv(name, default)

GEMINI_API_KEY = _get_secret_or_env("GEMINI_API_KEY")

# =========================
# CSS (consolidated)
# =========================

st.markdown("""
    <style>
    .main-header {
        font-size: 2.25rem;
        font-weight: 800;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #444;
        font-size: 1rem;
        margin-bottom: 1.25rem;
    }
    .citation-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 16px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 8px 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .citation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        border-left-color: #0d5aa7;
    }
    .citation-title {
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 8px;
        font-size: 14px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .citation-evidence {
        font-style: italic;
        color: #444;
        margin-bottom: 8px;
        background: #f0f4f8;
        padding: 10px;
        border-radius: 6px;
        border-left: 3px solid #28a745;
        font-size: 13px;
        line-height: 1.4;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .citation-card a {
        color: #1f77b4;
        text-decoration: none;
        transition: color 0.2s ease;
    }
    .citation-card a:hover {
        color: #0d5aa7;
        text-decoration: underline;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    /* Chat message styling */
    .stChatMessage {
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# Header with logo
# =========================

st.markdown('<div class="main-header">Internal PM Chat Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask questions across Confluence, Slack, Docs, Zendesk, and Jira</div>', unsafe_allow_html=True)

# =========================
# Sidebar
# =========================

with st.sidebar:
    st.header("Chat Configuration")

    st.info("""
    **Agent searches across:**
    - 📚 Docs (Qdrant)
    - 💬 Slack Messages
    - 📖 Confluence Pages
    - 🎫 Zendesk Tickets (Incorta)
    - 📋 Jira Issues (Incorta)
    """)

    st.markdown("---")
    st.subheader("System Info")

    # Show API key rotation info if available
    api_key_info = "Single API Key"
    if st.session_state.get("agent_executor") is not None:
        executor = st.session_state["agent_executor"]
        # Check if it's a RetryAgentExecutor with api_manager
        if hasattr(executor, 'api_manager'):
            num_keys = len(executor.api_manager.api_keys)
            current_idx = executor.api_manager.current_index
            api_key_info = f"{num_keys} API Keys (Current: #{current_idx + 1})"

    st.info(f"""
    **AI Model:** {LLM_NAME}
    **Vector DB:** Qdrant
    **Search:** Multi-source analysis
    **API Keys:** {api_key_info}
    """)

    # Cache statistics
    st.markdown("---")
    st.subheader("Cache Statistics")
    cache_stats = get_cache_manager().get_stats()
    st.metric("Active Cached Items", cache_stats["active_items"])

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Cache", use_container_width=True):
            get_cache_manager().clear()
            st.success("Cache cleared!")
            st.rerun()

    with col2:
        if st.button("New Chat", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["chat_history"] = []
            st.success("Chat reset!")
            st.rerun()

    st.markdown("---")
    st.subheader("Usage Tips")
    st.markdown("""
    - Ask follow-up questions naturally
    - Reference previous answers with "it", "that", etc.
    - Use specific keywords for best results
    - All sources searched automatically
    """)

    # Show conversation count
    if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
        st.markdown("---")
        st.metric("Messages in Chat", len(st.session_state["messages"]))

# =========================
# Session state initialization
# =========================

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "agent_executor" not in st.session_state:
    st.session_state["agent_executor"] = None

# =========================
# Guardrails
# =========================

def _ensure_gemini_key_if_needed():
    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY is not set. Add it to Streamlit secrets or your environment.")
        st.stop()

# =========================
# Helper Functions
# =========================

def render_sources(sources):
    """Render source citations grouped by type in expandable tabs."""
    if not sources:
        return
    
    def categorize_source(source):
        """Categorize a source into one of the 5 groups."""
        url = (source.get("url") or source.get("permalink") or "").lower()
        source_field = source.get("source", "").lower()
        
        # 1. Check for Slack
        if "channel" in source or "permalink" in source or source_field == "slack":
            return "slack"
        
        # 2. Check for Confluence
        if "confluence" in url or source_field == "confluence":
            return "confluence"
        
        # 3. Check for knowledge_base and subcategorize by URL
        if source_field == "knowledge_base" or (url.startswith("http") and source_field != "slack"):
            # Pattern matching for knowledge_base URLs
            if any(pattern in url for pattern in ["docs.incorta.com", "/docs/", "documentation"]):
                return "incorta_docs"
            elif any(pattern in url for pattern in ["community.incorta.com", "/community/"]):
                return "incorta_community"
            elif any(pattern in url for pattern in ["support.incorta.com", "/support/"]):
                return "incorta_support"
            else:
                # Skip sources that don't match any pattern (per preference #2)
                return None
        
        # 4. Check for Zendesk (support-related)
        if "zendesk" in url or source_field == "zendesk":
            return "incorta_support"
        
        # 5. Skip unknown sources (per preference #2)
        return None
    
    def format_date(source, category):
        """Format date from source based on category."""
        if category == "slack":
            # Slack has 'ts' (timestamp) or 'date' (formatted string)
            if "ts" in source:
                try:
                    ts = float(source.get("ts"))
                    dt = datetime.datetime.fromtimestamp(ts)
                    return dt.strftime("%b %d, %Y")
                except (ValueError, TypeError):
                    pass
            if "date" in source:
                date_str = source.get("date", "")
                if date_str and date_str != "Unknown date":
                    try:
                        # Try to parse if it's a formatted string
                        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                        return dt.strftime("%b %d, %Y")
                    except (ValueError, TypeError):
                        return date_str
            return None
        elif category == "confluence":
            # Confluence has 'last_modified'
            last_modified = source.get("last_modified", "")
            if last_modified and last_modified != "Recent":
                try:
                    # Try to parse ISO format or other formats
                    if isinstance(last_modified, str):
                        # Try common date formats
                        for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                            try:
                                dt = datetime.datetime.strptime(last_modified.split(".")[0], fmt)
                                return dt.strftime("%b %d, %Y")
                            except ValueError:
                                continue
                        return last_modified
                except (ValueError, TypeError):
                    pass
            return None
        else:
            # For docs, community, support - check for common date fields
            for field in ["date", "last_modified", "updated", "created"]:
                if field in source:
                    date_val = source.get(field)
                    if date_val:
                        try:
                            if isinstance(date_val, (int, float)):
                                dt = datetime.datetime.fromtimestamp(date_val)
                                return dt.strftime("%b %d, %Y")
                            elif isinstance(date_val, str):
                                for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                                    try:
                                        dt = datetime.datetime.strptime(date_val.split(".")[0], fmt)
                                        return dt.strftime("%b %d, %Y")
                                    except ValueError:
                                        continue
                                return date_val
                        except (ValueError, TypeError):
                            pass
            return None
    
    # Group sources by category
    grouped_sources = {
        "slack": [],
        "confluence": [],
        "incorta_docs": [],
        "incorta_community": [],
        "incorta_support": []
    }
    
    # Deduplicate and group (skip sources that return None from categorization)
    seen_urls = set()
    for source in sources:
        category = categorize_source(source)
        if category is None:  # Skip sources that don't match any pattern
            continue
        
        url = source.get("url") or source.get("permalink") or ""
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        
        grouped_sources[category].append(source)
    
    # Sort each group by score (highest first) - per preference #3
    def sort_by_score(sources):
        return sorted(sources, key=lambda s: float(s.get("score", 0.0)), reverse=True)
    
    for category in grouped_sources:
        grouped_sources[category] = sort_by_score(grouped_sources[category])
    
    # Category labels and icons
    category_config = {
        "slack": {"label": "Slack", "icon": "💬"},
        "confluence": {"label": "Confluence", "icon": "📖"},
        "incorta_docs": {"label": "Incorta Docs", "icon": "📚"},
        "incorta_community": {"label": "Incorta Community", "icon": "👥"},
        "incorta_support": {"label": "Incorta Support", "icon": "🎫"}
    }
    
    # Calculate total sources (only from non-empty categories)
    total_sources = sum(len(sources) for sources in grouped_sources.values())
    if total_sources == 0:
        return
    
    # Display grouped sources
    st.markdown(f"### 📚 Sources ({total_sources})")
    
    # Create expandable sections for each category (only show non-empty ones - per preference #4)
    for category, category_sources in grouped_sources.items():
        if not category_sources:  # Hide empty categories
            continue
        
        config = category_config[category]
        count = len(category_sources)
        
        with st.expander(f"{config['icon']} {config['label']} ({count})", expanded=False):
            for i, source in enumerate(category_sources, 1):
                # Determine title
                if category == "slack":
                    channel = source.get("channel", "unknown")
                    username = source.get("username", "unknown")
                    title = f"#{channel} - @{username}"
                else:
                    title = source.get("title", "Untitled")
                
                # Build URL
                url = source.get("url") or source.get("permalink") or ""
                
                # Format date
                date_str = format_date(source, category)
                date_display = f' <small style="color: #888;">• {date_str}</small>' if date_str else ""
                
                # Build evidence/text snippet
                evidence = source.get("text", "") or source.get("excerpt", "")
                if len(evidence) > 200:
                    evidence = evidence[:200] + "..."
                
                # Render citation card
                st.markdown(f"""
                <div class="citation-card">
                    <div class="citation-title">{config['icon']} {i}. {html.escape(title)}{date_display}</div>
                    {'<div class="citation-evidence">' + html.escape(evidence) + '</div>' if evidence else ''}
                    {'<small style="color: #666;">🔗 <a href="' + html.escape(url) + '" target="_blank">' + html.escape(url[:50] + ('...' if len(url) > 50 else '')) + '</a></small>' if url else ''}
                </div>
                """, unsafe_allow_html=True)


def build_conversation_context():
    """Build conversation context string for agent."""
    if not st.session_state["chat_history"]:
        return ""

    # Get last 6 messages (3 exchanges)
    recent_history = st.session_state["chat_history"][-6:]
    context_lines = []

    for msg in recent_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:200]  # Truncate long messages
        context_lines.append(f"{role}: {content}")

    return "\n".join(context_lines)


def process_query(query: str):
    """Process a user query and return response with sources."""
    _ensure_gemini_key_if_needed()

    # Build conversation context
    conversation_context = build_conversation_context()

    # Enhanced query with context for follow-ups
    enhanced_query = query
    if conversation_context:
        enhanced_query = f"Previous conversation:\n{conversation_context}\n\nCurrent question: {query}"

    # Check cache for agentic search results
    cache_filters = {
        "mode": "agentic",
        "query": query,
        "context": conversation_context[:100]  # Include short context in cache key
    }
    cached_results = get_cached_search_results(query, cache_filters)

    if cached_results:
        return {
            "answer": cached_results.get("final_answer"),
            "sources": cached_results.get("all_sources", []),
            "tools_used": cached_results.get("tools_used", []),
            "from_cache": True
        }

    # Initialize agent executor if not exists
    if st.session_state["agent_executor"] is None:
        with st.spinner("🤖 Initializing agent..."):
            try:
                st.session_state["agent_executor"] = create_pm_agent(api_key=None)
            except Exception as e:
                st.error(f"Failed to initialize agent: {e}")
                st.stop()

    # Run agent
    agent_executor = st.session_state["agent_executor"]

    # Collect agent execution details
    final_answer = None
    all_sources = []
    tools_used = set()
    step_sources = []  # list of lists: sources per observation step

    try:
        # Collect all chunks
        all_chunks = []
        for chunk in agent_executor.stream({"input": enhanced_query}):
            all_chunks.append(chunk)

            # Collect tool usage
            if "actions" in chunk:
                for action in chunk["actions"]:
                    tools_used.add(action.tool)

            if "steps" in chunk:
                for step in chunk["steps"]:
                    # Extract sources from observations
                    try:
                        obs_data = step.observation
                        collected = []
                        if isinstance(obs_data, list):
                            for item in obs_data:
                                if isinstance(item, dict) and any(k in item for k in ["url", "title", "text", "permalink"]):
                                    all_sources.append(item)
                                    collected.append(item)
                        if collected:
                            step_sources.append(collected)
                    except:
                        pass

            if "output" in chunk:
                final_answer = chunk["output"]

        # Determine which sources were actually used in the answer
        import re
        
        def normalize_url(u: str) -> str:
            try:
                return (u or "").strip().lower()
            except:
                return (u or "").lower()

        def normalize_title(t: str) -> str:
            try:
                return (t or "").strip().lower()
            except:
                return (t or "").lower()

        # Extract keywords from the answer that might reference sources
        mentioned_sources = []
        if isinstance(final_answer, str) and final_answer:
            answer_lower = final_answer.lower()
            answer_words = set(answer_lower.split())
            
            # Find sources that match by:
            # 1. URL domain/partial match in answer
            # 2. Title keywords in answer
            # 3. Source-specific identifiers (ticket numbers, page titles, etc.)
            for s in all_sources:
                url = normalize_url(s.get("url") or s.get("permalink") or "")
                title = normalize_title(s.get("title") or "")
                
                # Check if URL domain or partial URL appears in answer
                if url:
                    # Extract domain or key parts
                    url_parts = url.replace("https://", "").replace("http://", "").split("/")
                    if url_parts and url_parts[0]:
                        domain = url_parts[0].split(".")[0]  # e.g., "confluence" from "confluence.incorta.com"
                        if domain in answer_lower or any(part in answer_lower for part in url_parts[:3] if part):
                            mentioned_sources.append(s)
                            continue
                
                # Check if title keywords appear in answer
                if title and len(title) > 3:
                    title_words = set(title.split())
                    # If 2+ title words appear in answer, likely referenced
                    common_words = title_words.intersection(answer_words)
                    if len(common_words) >= 2:
                        mentioned_sources.append(s)
                        continue
                
                # Check for Slack channel/username mentions
                if "channel" in s:
                    channel = s.get("channel", "").lower()
                    if channel and f"#{channel}" in answer_lower:
                        mentioned_sources.append(s)
                        continue
                
                # Check for ticket/issue numbers (Zendesk/Jira)
                if "ticket" in answer_lower or "issue" in answer_lower:
                    # Look for patterns like "PROD-123" or ticket IDs
                    ticket_patterns = re.findall(r'[A-Z]+-\d+|\d{5,}', answer_lower)
                    if ticket_patterns and url:
                        # If answer mentions ticket patterns and we have a URL, likely related
                        mentioned_sources.append(s)
                        continue

        # Use only sources that were actually mentioned/referenced
        used_sources = mentioned_sources if mentioned_sources else []

        # If still no sources, try a more lenient approach: use only the LAST observation step
        # (the one that directly led to the final answer)
        if not used_sources and step_sources:
            last_step_sources = step_sources[-1]
            # Deduplicate
            seen = set()
            used_sources = []
            for s in last_step_sources:
                key = (s.get("url") or s.get("permalink") or s.get("title") or "")
                if key and key not in seen:
                    seen.add(key)
                    used_sources.append(s)
            
        # Final fallback: if agent used tools, only show sources if we have very few
        # (likely all were used) - otherwise show nothing rather than wrong sources
        if not used_sources:
            if len(all_sources) <= 5:
                # If we have 5 or fewer sources, likely all were used
                seen = set()
                used_sources = []
                for s in all_sources:
                    key = (s.get("url") or s.get("permalink") or s.get("title") or "")
                    if key and key not in seen:
                        seen.add(key)
                        used_sources.append(s)
            else:
                # Too many sources, don't show any rather than showing wrong ones
                used_sources = []

        # Cache results
        try:
            cache_search_results(query, cache_filters, {
                "final_answer": final_answer,
                "all_sources": used_sources,
                "tools_used": list(tools_used),
                "mode": "agentic"
            })
        except Exception as e:
            pass  # Don't fail on cache errors

        return {
            "answer": final_answer,
            "sources": used_sources,
            "tools_used": list(tools_used),
            "from_cache": False
        }

    except Exception as e:
        st.error(f"Agent execution failed: {e}")
        return {
            "answer": f"Sorry, I encountered an error: {str(e)}",
            "sources": [],
            "tools_used": [],
            "from_cache": False
        }


# =========================
# Display chat history
# =========================

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Render sources if available
        if message.get("sources"):
            render_sources(message["sources"])

        # Show cache indicator
        if message.get("from_cache"):
            st.caption("📦 From cache")

# =========================
# Chat input
# =========================

if prompt := st.chat_input("Ask a question about your PM tools and processes..."):
    # Add user message to chat
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.session_state["chat_history"].append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking and searching..."):
            response = process_query(prompt)

        # Display answer
        st.markdown(response["answer"])

        # Display sources
        if response["sources"]:
            render_sources(response["sources"])

        # Show cache indicator
        if response["from_cache"]:
            st.caption("📦 From cache")

        # Add assistant message to chat
        st.session_state["messages"].append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"],
            "from_cache": response["from_cache"]
        })
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": response["answer"]
        })

# =========================
# Welcome message
# =========================

if len(st.session_state["messages"]) == 0:
    st.markdown("---")
    st.markdown("### 👋 Welcome! Ask me anything about:")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **📚 Documentation**
        - Product features
        - Configuration guides
        - Best practices
        """)

    with col2:
        st.markdown("""
        **🎫 Customer Issues**
        - Zendesk tickets
        - Common problems
        - Support patterns
        """)

    with col3:
        st.markdown("""
        **📋 Development**
        - Jira tickets
        - Roadmap items
        - Feature status
        """)

    st.markdown("---")
    st.markdown("**Example questions:**")
    st.markdown("""
    - "What is SAML authentication in Incorta?"
    - "Show me recent Zendesk tickets about performance issues"
    - "What's the status of the new dashboard feature in Jira?"
    - "How do I configure materialized views?"
    """)

# =========================
# Footer
# =========================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <small>Internal PM Chat • Conversational Search Assistant • Powered by AI</small>
</div>
""", unsafe_allow_html=True)
