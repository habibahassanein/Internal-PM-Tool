import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from dotenv import load_dotenv, find_dotenv

# Load environment variables
try:
    load_dotenv()
except AssertionError:
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = "docs"
VECTOR_NAME = "content_vector"

# Configuration
CSV_PATH = "data/cleaned_pages.csv"
BATCH_SIZE = 100  # Upload in batches of 100
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # 768-dim model

def upload_vectors():
    # Load data
    print(f"Loading data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} pages")
    
    # Initialize embedding model
    print(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {embedding_dim}")
    
    # Connect to Qdrant
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Prepare texts for embedding (use title + text for richer context)
    texts = []
    for idx, row in df.iterrows():
        title = row.get("title", "") or ""
        content = row.get("cleaned_content", "") or ""
        # Combine title and content for embedding
        combined_text = f"{title}\n\n{content}".strip()
        texts.append(combined_text)
    
    # Generate embeddings in batches
    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )
    
    # Upload to Qdrant in batches
    print(f"Uploading {len(embeddings)} vectors to Qdrant in batches of {BATCH_SIZE}...")
    total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in tqdm(range(0, len(df), BATCH_SIZE), total=total_batches, desc="Uploading"):
        batch_df = df.iloc[i:i + BATCH_SIZE]
        batch_embeddings = embeddings[i:i + BATCH_SIZE]
        
        points = []
        for idx, (row_idx, row) in enumerate(batch_df.iterrows()):
            point = PointStruct(
                id=row_idx,  # Use DataFrame index as unique ID
                vector={VECTOR_NAME: batch_embeddings[idx].tolist()},
                payload={
                    "url": row.get("url", ""),
                    "title": row.get("title", ""),
                    "text": row.get("cleaned_content", "")[:10000],  # Limit text to 10k chars for payload
                }
            )
            points.append(point)
        
        # Upload batch
        client.upsert(
            collection_name=COLLECTION,
            points=points
        )
    
    print(f"✓ Successfully uploaded {len(df)} vectors to collection '{COLLECTION}'")
    
    # Verify upload
    collection_info = client.get_collection(COLLECTION)
    print(f"Collection now contains {collection_info.points_count} points")

if __name__ == "__main__":
    upload_vectors()
