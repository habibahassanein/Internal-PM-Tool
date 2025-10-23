import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from src.handler.gemini_handler import answer_with_citations, answer_with_multiple_sources
from src.handler.slack_handler import search_slack_messages
from src.handler.confluence_handler import search_confluence_pages

# Load environment
load_dotenv()

# Configuration - Use Streamlit secrets if available (for cloud deployment), otherwise fall back to env vars
if "QDRANT_URL" in st.secrets:
    QDRANT_URL = st.secrets["QDRANT_URL"]
    QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY", "")
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

COLLECTION = "docs"
VECTOR_NAME = "content_vector"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
TOP_K = 5

# Logo path (place your company logo here)
LOGO_PATH = "assets/company_logo.png"  # Create assets/ folder and add logo

# Initialize models (cache to avoid reloading)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource
def get_qdrant_client():
    """Initialize Qdrant client with error handling"""
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        # Test the connection
        client.get_collections()
        return client
    except Exception as e:
        st.error(f"Failed to connect to Qdrant: {str(e)}")
        st.info(f"Qdrant URL: {QDRANT_URL}")
        raise

# Page config
st.set_page_config(
    page_title="Internal PM Tool Search Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .search-box {
        margin-bottom: 2rem;
    }
    .citation-card {
        background: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .citation-title {
        font-weight: 600;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .citation-evidence {
        font-style: italic;
        color: #555;
        margin: 0.5rem 0;
    }
    .answer-box {
        background: #e8f4f8;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
    }
    .source-chip {
        display: inline-block;
        background: #1f77b4;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        font-size: 0.85rem;
        margin: 0.2rem;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Header with logo
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if Path(LOGO_PATH).exists():
        # Center the logo using columns
        logo_col1, logo_col2, logo_col3 = st.columns([1, 2, 1])
        with logo_col2:
            st.image(LOGO_PATH, width=200)
    st.markdown('<div class="main-header">Internal PM Tool Search Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">🔍 Powered by AI • Search across docs, community, and support</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    num_results = st.slider("Number of results", min_value=3, max_value=10, value=5, step=1)
    
    st.markdown("---")
    st.subheader("🔍 Search Sources")
    
    # Search source toggles
    search_knowledge_base = st.checkbox("📚 Knowledge Base (Qdrant)", value=True, help="Search your indexed documents")
    search_slack = st.checkbox("💬 Slack Messages", value=False, help="Search Slack channels and messages")
    search_confluence = st.checkbox("📖 Confluence Pages", value=False, help="Search Confluence documentation")
    
    # Slack-specific filters
    if search_slack:
        slack_channel = st.text_input("Slack Channel (optional)", placeholder="e.g., general, engineering", help="Leave empty to search all channels")
        slack_max_age = st.selectbox("Message Age", ["24h", "7d", "30d", "90d", "All"], index=2, help="Maximum age of messages to search")
    
    # Confluence-specific filters  
    if search_confluence:
        confluence_space = st.text_input("Confluence Space (optional)", placeholder="e.g., PROJ, DOCS", help="Leave empty to search all spaces")
    
    st.markdown("---")
    st.subheader("📊 System Info")
    st.info(f"""
    **Vector DB:** Qdrant  
    **Embedding Model:** MPNet-base-v2  
    **LLM:** Gemini 2.5 Flash  
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

# Main search interface
query = st.text_input(
    "🔎 Ask a question about Incorta, product features, or troubleshooting:",
    placeholder="SAML SSO?",
    key="search_query"
)

search_button = st.button("🚀 Search", use_container_width=True)

if search_button and query:
    # Initialize results
    qdrant_results = []
    slack_results = []
    confluence_results = []
    
    # Search Knowledge Base (Qdrant)
    if search_knowledge_base:
        with st.spinner("🔄 Searching knowledge base..."):
            try:
                # Load models
                model = load_embedding_model()
                client = get_qdrant_client()

                # Generate query embedding
                query_vector = model.encode(query, normalize_embeddings=True).tolist()

                # Search Qdrant
                results = client.search(
                    collection_name=COLLECTION,
                    query_vector=(VECTOR_NAME, query_vector),
                    limit=num_results,
                    with_payload=True
                )
                
                # Prepare Qdrant results
                for r in results:
                    qdrant_results.append({
                        "title": r.payload.get("title", ""),
                        "url": r.payload.get("url", ""),
                        "text": r.payload.get("text", ""),
                        "score": r.score
                    })
                    
            except Exception as e:
                st.error("❌ Knowledge base search failed!")
                st.exception(e)
                st.markdown("### Troubleshooting Tips:")
                st.markdown("""
                1. **Check Qdrant Connection**: Ensure your Qdrant instance is accessible
                2. **Verify Secrets**: In Streamlit Cloud, go to 'Manage app' → 'Settings' → 'Secrets' and add:
                   ```toml
                   QDRANT_URL = "your-qdrant-url"
                   QDRANT_API_KEY = "your-api-key"
                   GEMINI_API_KEY = "your-gemini-key"
                   ```
                3. **Check Collection**: Verify the collection '{COLLECTION}' exists in your Qdrant instance
                4. **Network Access**: Ensure Qdrant URL is publicly accessible or properly configured
                """)
                if not search_slack and not search_confluence:
                    st.stop()
    
    # Search Slack
    if search_slack:
        with st.spinner("💬 Searching Slack messages..."):
            try:
                # Convert age selection to hours
                age_mapping = {"24h": 24, "7d": 168, "30d": 720, "90d": 2160, "All": 0}
                max_age_hours = age_mapping.get(slack_max_age, 720)
                
                slack_results = search_slack_messages(
                    query=query,
                    max_results=num_results,
                    channel_filter=slack_channel if slack_channel else None,
                    max_age_hours=max_age_hours
                )
                
            except Exception as e:
                st.error("❌ Slack search failed!")
                st.exception(e)
                st.info("💡 Make sure SLACK_USER_TOKEN is set in your Streamlit secrets")
    
    # Search Confluence
    if search_confluence:
        with st.spinner("📖 Searching Confluence pages..."):
            try:
                confluence_results = search_confluence_pages(
                    query=query,
                    max_results=num_results,
                    space_key=confluence_space if confluence_space else None
                )
                
            except Exception as e:
                st.error("❌ Confluence search failed!")
                st.exception(e)
                st.info("💡 Make sure CONFLUENCE_URL, CONFLUENCE_EMAIL, and CONFLUENCE_API_TOKEN are set in your Streamlit secrets")
    
    # Check if we have any results
    total_results = len(qdrant_results) + len(slack_results) + len(confluence_results)
    
    if total_results == 0:
        st.warning("No results found from any source. Try a different query or enable more search sources.")
    else:
        # Generate AI answer using multiple sources
        with st.spinner("🤖 Generating answer..."):
            if search_knowledge_base and (search_slack or search_confluence):
                # Use multiple sources
                response = answer_with_multiple_sources(query, qdrant_results, slack_results, confluence_results)
            elif search_knowledge_base:
                # Use only Qdrant results
                passages = [{"title": r.get("title", ""), "url": r.get("url", ""), "text": r.get("text", "")} for r in qdrant_results]
                response = answer_with_citations(query, passages)
            else:
                # Use only Slack/Confluence results
                all_passages = []
                for r in slack_results + confluence_results:
                    all_passages.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "text": r.get("text", "") or r.get("excerpt", "")
                    })
                response = answer_with_citations(query, all_passages)
        
        # Display answer
        st.markdown("---")
        st.subheader("📝 Answer")
        
        if response.get("exists"):
            st.markdown(f'<div class="answer-box">{response.get("answer", "")}</div>', unsafe_allow_html=True)
            
            # Display citations
            citations = response.get("citations", [])
            if citations:
                st.markdown("### 📚 Sources")
                for i, cite in enumerate(citations, 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="citation-card">
                            <div class="citation-title">{i}. {cite.get('title', 'Untitled')}</div>
                            <div class="citation-evidence">"{cite.get('evidence', '')}"</div>
                            <a href="{cite.get('url', '#')}" target="_blank">🔗 View source</a>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ The available sources don't contain sufficient information to answer this query.")
            st.info(response.get("answer", ""))
        
        # Show raw results in expanders
        if qdrant_results:
            with st.expander(f"📚 Knowledge Base Results ({len(qdrant_results)} found)"):
                for i, r in enumerate(qdrant_results, 1):
                    st.markdown(f"**{i}. {r.get('title', 'Untitled')}** (Score: {r.get('score', 0):.4f})")
                    st.markdown(f"🔗 [{r.get('url', '')}]({r.get('url', '')})")
                    st.markdown(f"_{r.get('text', '')[:300]}..._")
                    st.markdown("---")
        
        if slack_results:
            with st.expander(f"💬 Slack Messages ({len(slack_results)} found)"):
                for i, r in enumerate(slack_results, 1):
                    st.markdown(f"**{i}. #{r.get('channel', 'unknown')} - @{r.get('username', 'unknown')}**")
                    st.markdown(f"🔗 [{r.get('permalink', '')}]({r.get('permalink', '')})")
                    st.markdown(f"_{r.get('text', '')[:300]}..._")
                    st.markdown("---")
        
        if confluence_results:
            with st.expander(f"📖 Confluence Pages ({len(confluence_results)} found)"):
                for i, r in enumerate(confluence_results, 1):
                    st.markdown(f"**{i}. {r.get('title', 'Untitled')}** ({r.get('space', 'Unknown')})")
                    st.markdown(f"🔗 [{r.get('url', '')}]({r.get('url', '')})")
                    st.markdown(f"_{r.get('excerpt', '')[:300]}..._")
                    st.markdown("---")

# Example queries
if not query:
    st.markdown("---")
    st.subheader("💬 Example Queries")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("🔐 How do I set up SAML authentication?")
        st.info("📊 What are materialized views in Incorta?")
    
    with col2:
        st.info("🐛 Troubleshooting data agent connection errors")
        st.info("⚡ How to optimize query performance?")
    
    with col3:
        st.info("💬 Recent discussions about API issues")
        st.info("📖 Documentation on deployment process")


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <small>Internal PM Tool Search Assistant • Built with Streamlit, Qdrant, and Google Gemini</small>
</div>
""", unsafe_allow_html=True)

