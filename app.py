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
    page_title="Internal PM Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants / Tunables
LLM_NAME = "Gemini 2.5 Flash Lite Preview"

# =========================
# Secrets & Keys
# =========================

def _get_secret_or_env(name: str, default: str = "") -> str:
    if name in st.secrets:
        return st.secrets.get(name, default)
    return os.getenv(name, default)

GEMINI_API_KEY = _get_secret_or_env("GEMINI_API_KEY")

# =========================
# Cached resources
# =========================


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
    .answer-box {
        background: linear-gradient(135deg, #f0f2f6 0%, #e8f0fe 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #1f77b4;
        margin: 12px 0;
        line-height: 1.7;
        box-shadow: 0 4px 6px rgba(0,0,0,0.08);
    }
    .citation-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #e1e5e9;
        margin: 10px 0;
        box-shadow: 0 3px 6px rgba(0,0,0,0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .citation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.12);
    }
    .citation-title {
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 8px;
        font-size: 15px;
    }
    .citation-evidence {
        font-style: italic;
        color: #333;
        margin-bottom: 8px;
        background: #f8f9fa;
        padding: 10px;
        border-radius: 6px;
        border-left: 3px solid #28a745;
        font-size: 13px;
        line-height: 1.45;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .search-summary {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 14px;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 10px 0;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# Header with logo
# =========================

st.markdown('<div class="main-header">Internal PM Tool</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Comprehensive Search Across All Sources</div>', unsafe_allow_html=True)

# =========================
# Sidebar
# =========================

with st.sidebar:
    st.header("Search Configuration")
    
    st.info("""
    **Agent will intelligently search:**
    - 📚 Docs (Qdrant)
    - 💬 Slack Messages
    - 📖 Confluence Pages
    - 🎫 Zendesk Tickets (Incorta)
    - 📋 Jira Issues (Incorta)
    """)
    
    st.markdown("---")
    st.subheader("System Info")
    st.info(f"""
    **AI Model:** {LLM_NAME}  
    **Vector DB:** Qdrant  
    **Search:** Multi-source analysis
    """)
    
    # Cache statistics
    st.markdown("---")
    st.subheader("Cache Statistics")
    cache_stats = get_cache_manager().get_stats()
    st.metric("Active Cached Items", cache_stats["active_items"])
    if st.button("Clear Cache"):
        get_cache_manager().clear()
        st.success("Cache cleared!")
        st.rerun()
    
    st.markdown("---")
    st.subheader("Usage Tips")
    st.markdown("""
    - Ask specific questions using keyterms for best results
    - Use natural language with technical keywords
    - Results are ranked by relevance
    - All sources searched automatically
    - **Note**: This app works best with keyterm searching and is still under development
    """)

# =========================
# Session state helpers
# =========================

if "search_query" not in st.session_state:
    st.session_state["search_query"] = ""
if "agent_executor" not in st.session_state:
    st.session_state["agent_executor"] = None

# =========================
# Main Search Form
# =========================

with st.form("search_form", clear_on_submit=False):
    query = st.text_input(
        "Search Query",
        placeholder="Ask a question about Incorta, product features, or troubleshooting...",
        key="search_query",
        help="Search across all available sources automatically"
    )
    submitted = st.form_submit_button("Search All Sources", use_container_width=True)

# =========================
# Guardrails
# =========================

def _ensure_gemini_key_if_needed():
    # Only needed if we are going to call the LLM (i.e., when there are any results)
    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY is not set. Add it to Streamlit secrets or your environment.")
        st.stop()

# =========================
# Search Logic
# =========================


# =========================
# Execute Search
# =========================

if submitted and query:
    _ensure_gemini_key_if_needed()
    
    # Check cache for agentic search results
    cache_filters = {
        "mode": "agentic",
        "query": query
    }
    cached_results = get_cached_search_results(query, cache_filters)
    
    if cached_results:
        final_answer = cached_results.get("final_answer")
        agent_steps = cached_results.get("agent_steps", [])
        tools_used = set(cached_results.get("tools_used", []))
        st.info("📦 Using cached agentic search results")
    else:
        # Initialize agent executor if not exists (will use API manager with all keys if api_key=None)
        if st.session_state["agent_executor"] is None:
            with st.spinner("🤖 Initializing agent..."):
                try:
                    # Pass None to use API manager with all keys from environment (GEMINI_API_KEY_1, _2, _3, etc.)
                    # Falls back to GEMINI_API_KEY if numbered keys not found
                    st.session_state["agent_executor"] = create_pm_agent(api_key=None)
                except Exception as e:
                    st.error(f"Failed to initialize agent: {e}")
                    st.stop()
        
        # Run agent
        agent_executor = st.session_state["agent_executor"]
        
        # Collect agent execution details
        final_answer = None
        agent_steps = []
        tools_used = set()
        
        with st.spinner("🤖 Agent is thinking and searching..."):
            try:
                # Collect all chunks first (Streamlit doesn't handle incremental updates well in expanders)
                all_chunks = []
                for chunk in agent_executor.stream({"input": query}):
                    all_chunks.append(chunk)
                    
                    # Collect step information
                    if "actions" in chunk:
                        for action in chunk["actions"]:
                            step_info = {
                                "type": "action",
                                "tool": action.tool,
                                "input": action.tool_input
                            }
                            agent_steps.append(step_info)
                            tools_used.add(action.tool)
                    
                    if "steps" in chunk:
                        for step in chunk["steps"]:
                            step_info = {
                                "type": "observation",
                                "tool": step.action.tool,
                                "observation": str(step.observation)
                            }
                            agent_steps.append(step_info)
                    
                    if "output" in chunk:
                        final_answer = chunk["output"]
                
                # Cache agent execution results
                try:
                    cache_search_results(query, cache_filters, {
                        "final_answer": final_answer,
                        "agent_steps": agent_steps,
                        "tools_used": list(tools_used),
                        "mode": "agentic"
                    })
                except Exception as e:
                    # Log but don't fail on cache errors
                    pass
            
            except Exception as e:
                st.error(f"Agent execution failed: {e}")
                with st.expander("Error Details"):
                    st.exception(e)
                st.stop()
    
    st.markdown("---")
    st.subheader("Agent Execution")
    
    # Display final answer
    if final_answer:
        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        st.markdown(final_answer)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### Agent Summary")
        st.markdown(f"""
        <div class="search-summary">
            <h4>Agent Analysis</h4>
            <p><strong>Query:</strong> "{html.escape(query)}"</p>
            <p><strong>Mode:</strong> Agentic Search</p>
            <p><strong>Tools Used:</strong> {len(tools_used)} ({', '.join(sorted(tools_used))})</p>
            <p><strong>Total Steps:</strong> {len(agent_steps)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display agent thought process in expander
        with st.expander("🔍 Agent Thought Process", expanded=False):
            for i, step in enumerate(agent_steps, 1):
                if step["type"] == "action":
                    st.markdown(f"**Step {i}: Tool Call**")
                    st.markdown(f"- **Tool:** `{step['tool']}`")
                    st.code(str(step['input']), language=None)
                elif step["type"] == "observation":
                    st.markdown(f"**Step {i}: Tool Output**")
                    st.markdown(f"- **Tool:** `{step['tool']}`")
                    # Truncate long observations for display
                    obs = str(step['observation'])
                    if len(obs) > 1000:
                        st.text(obs[:1000] + "\n... (truncated)")
                    else:
                        st.text(obs)
                st.markdown("---")
    else:
        st.warning("Agent did not return a final answer.")
        # Still show steps if available
        if agent_steps:
            with st.expander("🔍 Agent Thought Process", expanded=True):
                for i, step in enumerate(agent_steps, 1):
                    if step["type"] == "action":
                        st.markdown(f"**Step {i}: Tool Call**")
                        st.markdown(f"- **Tool:** `{step['tool']}`")
                        st.code(str(step['input']), language=None)
                    elif step["type"] == "observation":
                        st.markdown(f"**Step {i}: Tool Output**")
                        st.markdown(f"- **Tool:** `{step['tool']}`")
                        obs = str(step['observation'])
                        if len(obs) > 1000:
                            st.text(obs[:1000] + "\n... (truncated)")
                        else:
                            st.text(obs)
                    st.markdown("---")

# =========================
# Example Queries
# =========================

if not query:
    st.markdown("---")
    st.subheader("Example Queries")
    
    st.markdown("""
    **Try these example queries:**
    
    🔐 How do I set up SAML authentication?  
    📊 What are materialized views in Incorta?  
    🐛 Troubleshooting data agent connection errors  
    ⚡ How to optimize query performance?  
    💬 Recent discussions about API issues  
    📖 Documentation on deployment process  
    """)

# =========================
# Footer
# =========================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1.25rem;">
    <small>Internal PM Tool • Comprehensive Search Assistant • Powered by AI</small>
</div>
""", unsafe_allow_html=True)
 