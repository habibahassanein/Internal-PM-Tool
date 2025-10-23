import os
import html
from pathlib import Path

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
    page_title="Internal PM Tool Search Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants / Tunables
COLLECTION = "docs"
VECTOR_NAME = "content_vector"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
QDRANT_SCORE_MIN = 0.25  # minimum cosine score to consider a hit
LOGO_PATH = "assets/company_logo.png"
LLM_NAME = "Gemini"  # surfaced in UI (handlers may choose the exact sub-model internally)

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
    return SentenceTransformer(MODEL_NAME)

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

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if Path(LOGO_PATH).exists():
        logo_col1, logo_col2, logo_col3 = st.columns([1, 2, 1])
        with logo_col2:
            st.image(LOGO_PATH, width=180)
    st.markdown('<div class="main-header">Internal PM Tool Search Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">🔍 Powered by AI • Search across docs, community, and support</div>', unsafe_allow_html=True)

# =========================
# Sidebar
# =========================

with st.sidebar:
    st.header("⚙️ Settings")

    num_results = st.slider("Number of results", min_value=3, max_value=10, value=5, step=1)

    st.markdown("---")
    st.subheader("🔍 Search Sources")

    search_knowledge_base = st.checkbox("📚 Knowledge Base (Qdrant)", value=True, help="Search your indexed documents")
    search_slack = st.checkbox("💬 Slack Messages", value=False, help="Search Slack channels and messages")
    search_confluence = st.checkbox("📖 Confluence Pages", value=False, help="Search Confluence documentation")

    if search_slack:
        slack_channel = st.text_input("Slack Channel (optional)", placeholder="e.g., general, engineering",
                                      help="Leave empty to search all channels")
        slack_max_age = st.selectbox("Message Age", ["24h", "7d", "30d", "90d", "All"], index=2,
                                     help="Maximum age of messages to search")
    else:
        slack_channel, slack_max_age = None, "30d"

    if search_confluence:
        confluence_space = st.text_input("Confluence Space (optional)", placeholder="e.g., PROJ, DOCS",
                                         help="Leave empty to search all spaces")
    else:
        confluence_space = None

    st.markdown("---")
    st.subheader("📊 System Info")
    st.info(f"""
**Vector DB:** Qdrant  
**Embedding Model:** MPNet-base-v2  
**LLM:** {LLM_NAME}  
**Collection:** {COLLECTION}
""")

    st.markdown("---")
    st.subheader("💡 Tips")
    st.markdown("""
- Use natural language queries  
- Be specific for better results  
- Check citations for source verification  
- Enable multiple sources for comprehensive results  
""")

# =========================
# Session state helpers (for clickable examples)
# =========================

if "search_query" not in st.session_state:
    st.session_state["search_query"] = ""

if "auto_submit" not in st.session_state:
    st.session_state["auto_submit"] = False

def _trigger_search_with_query(q: str):
    st.session_state["search_query"] = q
    st.session_state["auto_submit"] = True
    st.rerun()

# =========================
# Main Search Form
# =========================

with st.form("search_form", clear_on_submit=False):
    query = st.text_input(
        "🔎 Ask a question about Incorta, product features, or troubleshooting:",
        placeholder="SAML SSO?",
        key="search_query",
    )
    submitted = st.form_submit_button("🚀 Search", use_container_width=True)

# Auto-submit if user clicked a suggested query button
if st.session_state.get("auto_submit", False):
    submitted = True
    st.session_state["auto_submit"] = False  # reset flag

# =========================
# Guardrails
# =========================

def _ensure_sources_selected():
    if not any([search_knowledge_base, search_slack, search_confluence]):
        st.warning("Please enable at least one source in the sidebar.")
        st.stop()

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
        return float(score) >= QDRANT_SCORE_MIN
    except Exception:
        return False

def _confidence_label(source_count: int, total_results: int) -> tuple[str, str]:
    score = 0
    if search_knowledge_base:
        score += 30
    if search_slack:
        score += 25
    if search_confluence:
        score += 25
    if total_results >= 5:
        score += 20
    level = "High" if score >= 70 else ("Medium" if score >= 40 else "Low")
    color = "🟢" if score >= 70 else ("🟡" if score >= 40 else "🔴")
    return level, color

# =========================
# Execute Search
# =========================

if submitted and query:
    _ensure_sources_selected()

    # Initialize results
    qdrant_results = []
    slack_results = []
    confluence_results = []

    # Knowledge Base (Qdrant)
    if search_knowledge_base:
        with st.spinner("🔄 Searching knowledge base..."):
            try:
                model = load_embedding_model()
                client = get_qdrant_client()

                query_vector = model.encode(query, normalize_embeddings=True).tolist()

                results = client.search(
                    collection_name=COLLECTION,
                    query_vector=(VECTOR_NAME, query_vector),
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
                st.error("❌ Knowledge base search failed.")
                with st.expander("Troubleshooting (developer)"):
                    st.exception(e)
                    st.markdown(f"""
1. Check Qdrant connection/URL/API key.
2. Verify the collection '{COLLECTION}' exists and the vector name '{VECTOR_NAME}' is configured.
3. Ensure network access to the Qdrant endpoint.
""")
                # If no other sources are enabled, stop here
                if not (search_slack or search_confluence):
                    st.stop()

    # Slack
    if search_slack:
        with st.spinner("💬 Searching Slack messages..."):
            try:
                age_mapping = {"24h": 24, "7d": 168, "30d": 720, "90d": 2160, "All": 0}
                max_age_hours = age_mapping.get(slack_max_age, 720)

                slack_results = search_slack_messages(
                    query=query,
                    max_results=num_results,
                    channel_filter=slack_channel if slack_channel else None,
                    max_age_hours=max_age_hours
                ) or []

            except Exception as e:
                st.error("❌ Slack search failed.")
                with st.expander("Troubleshooting (developer)"):
                    st.exception(e)
                st.info("💡 Make sure SLACK_USER_TOKEN is set in Streamlit secrets.")

    # Confluence
    if search_confluence:
        with st.spinner("📖 Searching Confluence pages..."):
            try:
                confluence_results = search_confluence_pages(
                    query=query,
                    max_results=num_results,
                    space_key=confluence_space if confluence_space else None
                ) or []

            except Exception as e:
                st.error("❌ Confluence search failed.")
                with st.expander("Troubleshooting (developer)"):
                    st.exception(e)
                st.info("💡 Ensure CONFLUENCE_URL, CONFLUENCE_EMAIL, and CONFLUENCE_API_TOKEN are set.")

    total_results = len(qdrant_results) + len(slack_results) + len(confluence_results)

    if total_results == 0:
        st.warning("No results found from any source. Try a different query or enable more search sources.")
    else:
        # We will call the LLM, ensure key exists
        _ensure_gemini_key_if_needed()

        with st.spinner("🤖 Generating answer..."):
            if search_knowledge_base and (search_slack or search_confluence):
                response = answer_with_multiple_sources(query, qdrant_results, slack_results, confluence_results)
            elif search_knowledge_base:
                passages = [{"title": r.get("title", ""), "url": r.get("url", ""), "text": r.get("text", "")}
                            for r in qdrant_results]
                response = answer_with_citations(query, passages)
            else:
                all_passages = []
                for r in (slack_results + confluence_results):
                    all_passages.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "text": r.get("text", "") or r.get("excerpt", "")
                    })
                response = answer_with_citations(query, all_passages)

        st.markdown("---")
        st.subheader("📝 AI-Generated Answer")

        enabled_sources = []
        if search_knowledge_base:
            enabled_sources.append("Knowledge Base")
        if search_slack:
            enabled_sources.append("Slack")
        if search_confluence:
            enabled_sources.append("Confluence")

        st.markdown(f"""
        <div class="search-summary">
            <h4>🔍 Search Analysis</h4>
            <p><strong>Query:</strong> "{html.escape(query)}"</p>
            <p><strong>Sources Searched:</strong> {', '.join(enabled_sources) if enabled_sources else 'None'}</p>
            <p><strong>Total Results Found:</strong> {total_results}</p>
            <p><strong>Search Strategy:</strong> Multi-source AI analysis with relevance scoring</p>
        </div>
        """, unsafe_allow_html=True)

        if response.get("exists"):
            # Render answer as Markdown (avoid unsafe HTML for LLM output)
            st.markdown('<div class="answer-box">', unsafe_allow_html=True)
            st.markdown(response.get("answer", ""))
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("### 📊 Analysis Details")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("📚 Knowledge Base", len(qdrant_results),
                          delta=f"{len(qdrant_results)} documents analyzed" if qdrant_results else "No documents found")
            with col_b:
                st.metric("💬 Slack Messages", len(slack_results),
                          delta=f"{len(slack_results)} messages analyzed" if slack_results else "No messages found")
            with col_c:
                st.metric("📖 Confluence Pages", len(confluence_results),
                          delta=f"{len(confluence_results)} pages analyzed" if confluence_results else "No pages found")

            level, color = _confidence_label(len(enabled_sources), total_results)
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 15px; border-radius: 8px; border-left: 4px solid #28a745;">
                <strong>{color} Confidence Level: {level}</strong><br>
                <small>Based on source diversity ({len(enabled_sources)} sources) and result count ({total_results} results)</small>
            </div>
            """, unsafe_allow_html=True)

            # Citations
            citations = response.get("citations", [])
            if citations:
                st.markdown("### 📚 Supporting Sources")

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

                render_citation_block("📚", "Knowledge Base Sources", kb_citations)
                render_citation_block("💬", "Slack Discussion Sources", slack_citations)
                render_citation_block("📖", "Confluence Documentation Sources", confluence_citations)

            else:
                st.warning("⚠️ **Insufficient Information**: The available sources don't contain sufficient information to answer this query.")
                st.info(f"**AI Response**: {response.get('answer', '')}")

                st.markdown("### 💡 Suggestions for Better Results")
                suggestions = []
                if not search_slack:
                    suggestions.append("Enable Slack search to find recent discussions.")
                if not search_confluence:
                    suggestions.append("Enable Confluence search to find documentation.")
                if not search_knowledge_base:
                    suggestions.append("Enable Knowledge Base search to find indexed documents.")
                if suggestions:
                    for s in suggestions:
                        st.markdown(f"- {s}")
                else:
                    st.markdown("- Try rephrasing your question with different keywords.")
                    st.markdown("- Check if the information might be in a different source.")
                    st.markdown("- Consider searching for related topics.")

        else:
            st.warning("⚠️ **No definitive answer** could be generated from the retrieved content.")
            st.info(f"**AI Response**: {response.get('answer', '')}")

        # Raw results expanders
        if qdrant_results:
            with st.expander(f"📚 Knowledge Base Results ({len(qdrant_results)} found)"):
                for i, r in enumerate(qdrant_results, 1):
                    st.markdown(f"**{i}. {r.get('title', 'Untitled')}** (Score: {r.get('score', 0):.4f})")
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
# Clickable Example Queries
# =========================

if not query:
    st.markdown("---")
    st.subheader("💬 Example Queries")

    examples_col1, examples_col2, examples_col3 = st.columns(3)

    with examples_col1:
        if st.button("🔐 How do I set up SAML authentication?", key="ex_saml"):
            _trigger_search_with_query("How do I set up SAML authentication?")
        if st.button("📊 What are materialized views in Incorta?", key="ex_matviews"):
            _trigger_search_with_query("What are materialized views in Incorta?")

    with examples_col2:
        if st.button("🐛 Troubleshooting data agent connection errors", key="ex_agent"):
            _trigger_search_with_query("Troubleshooting data agent connection errors")
        if st.button("⚡ How to optimize query performance?", key="ex_perf"):
            _trigger_search_with_query("How to optimize query performance?")

    with examples_col3:
        if st.button("💬 Recent discussions about API issues", key="ex_api"):
            _trigger_search_with_query("Recent discussions about API issues")
        if st.button("📖 Documentation on deployment process", key="ex_deploy"):
            _trigger_search_with_query("Documentation on deployment process")

# =========================
# Footer
# =========================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1.25rem;">
    <small>Internal PM Tool Search Assistant • Built with Streamlit, Qdrant, and Google Gemini</small>
</div>
""", unsafe_allow_html=True)
 