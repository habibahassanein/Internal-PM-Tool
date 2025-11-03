import os
import html

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
    """Render source citations as cards."""
    if not sources:
        return

    st.markdown("**📚 Sources:**")

    # Deduplicate sources by URL/permalink
    seen_urls = set()
    unique_sources = []
    for source in sources:
        url = source.get("url") or source.get("permalink") or ""
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_sources.append(source)
        elif not url:
            unique_sources.append(source)

    # Display each source as a citation card
    for i, source in enumerate(unique_sources[:8], 1):  # Limit to 8 citations for chat
        # Determine source type
        source_type = source.get("source", "unknown")
        if "channel" in source or "permalink" in source:
            source_type = "slack"
        elif "confluence" in source.get("url", "").lower():
            source_type = "confluence"
        elif "zendesk" in source.get("url", "").lower():
            source_type = "zendesk"
        elif "jira" in source.get("url", "").lower():
            source_type = "jira"
        elif source.get("url", "").startswith("http"):
            source_type = "knowledge_base"

        # Source icon mapping
        source_icons = {
            "slack": "💬",
            "confluence": "📖",
            "zendesk": "🎫",
            "jira": "📋",
            "knowledge_base": "📚",
            "unknown": "📄"
        }

        icon = source_icons.get(source_type, "📄")

        # Build title
        if source_type == "slack":
            channel = source.get("channel", "unknown")
            username = source.get("username", "unknown")
            title = f"#{channel} - @{username}"
        else:
            title = source.get("title", "Untitled")

        # Build URL
        url = source.get("url") or source.get("permalink") or ""

        # Build evidence/text snippet
        evidence = source.get("text", "") or source.get("excerpt", "")
        if len(evidence) > 200:
            evidence = evidence[:200] + "..."

        # Render citation card
        st.markdown(f"""
        <div class="citation-card">
            <div class="citation-title">{icon} {i}. {html.escape(title)}</div>
            {'<div class="citation-evidence">' + html.escape(evidence) + '</div>' if evidence else ''}
            {'<small style="color: #666;">🔗 <a href="' + url + '" target="_blank">' + url[:50] + ('...' if len(url) > 50 else '') + '</a></small>' if url else ''}
            <br><small style="color: #999;">Source: {source_type.replace('_', ' ').title()}</small>
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
                        if isinstance(obs_data, list):
                            for item in obs_data:
                                if isinstance(item, dict) and any(k in item for k in ["url", "title", "text", "permalink"]):
                                    all_sources.append(item)
                    except:
                        pass

            if "output" in chunk:
                final_answer = chunk["output"]

        # Cache results
        try:
            cache_search_results(query, cache_filters, {
                "final_answer": final_answer,
                "all_sources": all_sources,
                "tools_used": list(tools_used),
                "mode": "agentic"
            })
        except Exception as e:
            pass  # Don't fail on cache errors

        return {
            "answer": final_answer,
            "sources": all_sources,
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
