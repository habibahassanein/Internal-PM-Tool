import os
import html

import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from src.handler.gemini_handler import answer_with_citations, answer_with_multiple_sources
from src.handler.slack_handler import search_slack_messages
from src.handler.confluence_handler import search_confluence_pages

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
LLM_NAME = "Gemini 2.5 Flash"

# =========================
# Secrets & Keys
# =========================

def _get_secret_or_env(name: str, default: str = "") -> str:
    if name in st.secrets:
        return st.secrets.get(name, default)
    return os.getenv(name, default)

QDRANT_URL = _get_secret_or_env("QDRANT_URL")
QDRANT_API_KEY = _get_secret_or_env("QDRANT_API_KEY", "")
GEMINI_API_KEY = _get_secret_or_env("GEMINI_API_KEY")

# =========================
# Cached resources
# =========================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

@st.cache_resource
def get_qdrant_client():
    """Initialize Qdrant client with error handling."""
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        client.get_collections()  # smoke test
        return client
    except Exception as e:
        st.error("Failed to connect to Qdrant.")
        with st.expander("Connection details (developer)"):
            st.code(f"QDRANT_URL={QDRANT_URL}\nError={repr(e)}")
        raise

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
    
    num_results = st.slider("Results per source", min_value=5, max_value=15, value=10, step=1)
    
    st.markdown("---")
    st.subheader("Search Sources")
    st.info("""
    **All sources are automatically searched:**
    - 📚 Incorta Community, Docs & Support
    - 💬 Slack Messages  
    - 📖 Confluence Pages
    """)
    
    st.markdown("---")
    st.subheader("System Info")
    st.info(f"""
    **AI Model:** {LLM_NAME}  
    **Vector DB:** Qdrant  
    **Search:** Multi-source analysis
    """)
    
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

def _ensure_sources_selected():
    # All sources are automatically enabled
    pass

def _ensure_gemini_key_if_needed():
    # Only needed if we are going to call the LLM (i.e., when there are any results)
    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY is not set. Add it to Streamlit secrets or your environment.")
        st.stop()

# =========================
# Search Logic
# =========================

def _cosine_score_ok(score: float) -> bool:
    try:
        return float(score) >= 0.25  # minimum cosine score to consider a hit
    except Exception:
        return False

# =========================
# Execute Search
# =========================

if submitted and query:
    _ensure_sources_selected()

    # Initialize results - all sources are automatically searched
    qdrant_results = []
    slack_results = []
    confluence_results = []

    # Incorta Community, Docs & Support - Always searched
    with st.spinner("🔄 Searching Incorta Community, Docs & Support..."):
        try:
            model = load_embedding_model()
            client = get_qdrant_client()

            query_vector = model.encode(query, normalize_embeddings=True).tolist()

            results = client.search(
                collection_name="docs",
                query_vector=("content_vector", query_vector),
                limit=num_results,
                with_payload=True
            )

            for r in results:
                if _cosine_score_ok(r.score):
                    qdrant_results.append({
                        "title": r.payload.get("title", "") or "",
                        "url": r.payload.get("url", "") or "",
                        "text": r.payload.get("text", "") or "",
                        "score": r.score
                    })

        except Exception as e:
            st.error("❌ Incorta Community, Docs & Support search failed.")
            with st.expander("Troubleshooting (developer)"):
                st.exception(e)
                st.markdown(f"""
1. Check Qdrant connection/URL/API key.
2. Verify the collection 'docs' exists and the vector name 'content_vector' is configured.
3. Ensure network access to the Qdrant endpoint.
""")

    # Slack - Always searched
    with st.spinner("💬 Searching Slack messages..."):
        try:
            slack_results = search_slack_messages(
                query=query,
                max_results=num_results,
                channel_filter=None,  # Search all channels
                max_age_hours=0  # Search all history
            ) or []

        except Exception as e:
            st.error("❌ Slack search failed.")
            with st.expander("Troubleshooting (developer)"):
                st.exception(e)
            st.info("💡 Make sure SLACK_USER_TOKEN is set in Streamlit secrets.")

    # Confluence - Always searched
    with st.spinner("📖 Searching Confluence pages..."):
        try:
            confluence_results = search_confluence_pages(
                query=query,
                max_results=num_results,
                space_key=None  # Search all spaces
            ) or []

        except Exception as e:
            st.error("❌ Confluence search failed.")
            with st.expander("Troubleshooting (developer)"):
                st.exception(e)
            st.info("💡 Ensure CONFLUENCE_URL, CONFLUENCE_EMAIL, and CONFLUENCE_API_TOKEN are set.")

    total_results = len(qdrant_results) + len(slack_results) + len(confluence_results)

    if total_results == 0:
        st.warning("No results found from any source. Try a different query.")
    else:
        # We will call the LLM, ensure key exists
        _ensure_gemini_key_if_needed()

        with st.spinner("🤖 Generating answer..."):
            # Always use multiple sources since all are searched
            response = answer_with_multiple_sources(query, qdrant_results, slack_results, confluence_results)

        st.markdown("---")
        st.subheader("Search Results")

        # Simple analysis summary
        st.markdown(f"""
        <div class="search-summary">
            <h4>Search Analysis</h4>
            <p><strong>Query:</strong> "{html.escape(query)}"</p>
            <p><strong>Sources Searched:</strong> Incorta Community/Docs/Support, Slack, Confluence</p>
            <p><strong>Total Results:</strong> {total_results}</p>
            <p><strong>Method:</strong> Multi-source AI analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        if response.get("exists"):
            # Render answer as Markdown (avoid unsafe HTML for LLM output)
            st.markdown('<div class="answer-box">', unsafe_allow_html=True)
            st.markdown(response.get("answer", ""))
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("### Results by Source")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("📚 Incorta Community/Docs/Support", len(qdrant_results),
                          delta=f"{len(qdrant_results)} documents" if qdrant_results else "No results")
            with col_b:
                st.metric("💬 Slack Messages", len(slack_results),
                          delta=f"{len(slack_results)} messages" if slack_results else "No results")
            with col_c:
                st.metric("📖 Confluence Pages", len(confluence_results),
                          delta=f"{len(confluence_results)} pages" if confluence_results else "No results")

            # Citations
            citations = response.get("citations", [])
            if citations:
                st.markdown("### Supporting Sources")

                def render_citation_block(header_emoji: str, header_text: str, items: list[dict]):
                    if not items:
                        return
                    st.markdown(f"#### {header_emoji} {header_text}")
                    for i, cite in enumerate(items, 1):
                        title = cite.get("title", "Untitled") or "Untitled"
                        url = cite.get("url", "#") or "#"
                        evidence = cite.get("evidence", "") or ""
                        # Escape evidence to prevent HTML injection
                        evidence_safe = html.escape(evidence)
                        st.markdown(f"""
                        <div class="citation-card">
                            <div class="citation-title">{i}. {html.escape(title)}</div>
                            <div class="citation-evidence">{evidence_safe}</div>
                            <a href="{html.escape(url)}" target="_blank">🔗 Open source</a>
                        </div>
                        """, unsafe_allow_html=True)

                kb_citations = [c for c in citations if c.get('source', '') == 'knowledge_base']
                slack_citations = [c for c in citations if c.get('source', '') == 'slack']
                confluence_citations = [c for c in citations if c.get('source', '') == 'confluence']

                render_citation_block("📚", "Incorta Community/Docs/Support Sources", kb_citations)
                render_citation_block("💬", "Slack Discussion Sources", slack_citations)
                render_citation_block("📖", "Confluence Documentation Sources", confluence_citations)

            else:
                st.warning("⚠️ **Insufficient Information**: The available sources don't contain sufficient information to answer this query.")
                st.info(f"**AI Response**: {response.get('answer', '')}")

                st.markdown("### 💡 Suggestions for Better Results")
                st.markdown("- Try rephrasing your question with different keywords.")
                st.markdown("- Check if the information might be in a different source.")
                st.markdown("- Consider searching for related topics.")

        else:
            st.warning("⚠️ **No definitive answer** could be generated from the retrieved content.")
            st.info(f"**AI Response**: {response.get('answer', '')}")

        # Raw results expanders
        if qdrant_results:
            with st.expander(f"📚 Incorta Community/Docs/Support Results ({len(qdrant_results)} found)"):
                for i, r in enumerate(qdrant_results, 1):
                    st.markdown(f"**{i}. {r.get('title', 'Untitled')}**")
                    if r.get('url'):
                        st.markdown(f"🔗 [{r.get('url', '')}]({r.get('url', '')})")
                    preview = (r.get('text') or "")[:300]
                    if preview:
                        st.caption(preview + "…")
                    st.markdown("---")

        if slack_results:
            with st.expander(f"💬 Slack Messages ({len(slack_results)} found)"):
                for i, r in enumerate(slack_results, 1):
                    channel = r.get('channel', 'unknown')
                    user = r.get('username', 'unknown')
                    st.markdown(f"**{i}. #{channel} — @{user}**")
                    if r.get('permalink'):
                        st.markdown(f"🔗 [{r.get('permalink', '')}]({r.get('permalink', '')})")
                    preview = (r.get('text') or "")[:300]
                    if preview:
                        st.caption(preview + "…")
                    st.markdown("---")

        if confluence_results:
            with st.expander(f"📖 Confluence Pages ({len(confluence_results)} found)"):
                for i, r in enumerate(confluence_results, 1):
                    space = r.get('space', 'Unknown')
                    st.markdown(f"**{i}. {r.get('title', 'Untitled')}** ({space})")
                    if r.get('url'):
                        st.markdown(f"🔗 [{r.get('url', '')}]({r.get('url', '')})")
                    preview = (r.get('excerpt') or "")[:300]
                    if preview:
                        st.caption(preview + "…")
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
 