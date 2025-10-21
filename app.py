import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from src.handler.gemini_handler import answer_with_citations

# Load environment
load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
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
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

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
    """)

# Main search interface
query = st.text_input(
    "🔎 Ask a question about Incorta, product features, or troubleshooting:",
    placeholder="SAML SSO?",
    key="search_query"
)

search_button = st.button("🚀 Search", use_container_width=True)

if search_button and query:
    with st.spinner("🔄 Searching knowledge base..."):
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
        
        if not results:
            st.warning("No results found. Try a different query.")
        else:
            # Prepare passages for Gemini
            passages = []
            for r in results:
                passages.append({
                    "title": r.payload.get("title", ""),
                    "url": r.payload.get("url", ""),
                    "text": r.payload.get("text", "")
                })
            
            # Get AI-generated answer
            with st.spinner("🤖 Generating answer..."):
                response = answer_with_citations(query, passages)
            
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
                st.warning("⚠️ The available documents don't contain sufficient information to answer this query.")
                st.info(response.get("answer", ""))
            
            # Show raw results in expander
            with st.expander("🔍 View all retrieved documents"):
                for i, r in enumerate(results, 1):
                    st.markdown(f"**{i}. {r.payload.get('title', 'Untitled')}** (Score: {r.score:.4f})")
                    st.markdown(f"🔗 [{r.payload.get('url', '')}]({r.payload.get('url', '')})")
                    st.markdown(f"_{r.payload.get('text', '')[:300]}..._")
                    st.markdown("---")

# Example queries
if not query:
    st.markdown("---")
    st.subheader("💬 Example Queries")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("🔐 How do I set up SAML authentication?")
        st.info("📊 What are materialized views in Incorta?")
    
    with col2:
        st.info("🐛 Troubleshooting data agent connection errors")
        st.info("⚡ How to optimize query performance?")


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <small>Internal PM Tool Search Assistant • Built with Streamlit, Qdrant, and Google Gemini</small>
</div>
""", unsafe_allow_html=True)

