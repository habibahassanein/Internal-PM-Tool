# src/llm/gemini.py
import os, json, textwrap
import google.generativeai as genai
from dotenv import load_dotenv

# Load API key from environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

assert API_KEY, "Missing GEMINI_API_KEY in environment variables"
genai.configure(api_key=API_KEY)

# Model: Gemini 2.5 Flash (fast, cheap). Change if you prefer Pro later.
MODEL_NAME = "gemini-2.5-flash"


SYSTEM_MSG = ("""
You are Ibn Battouta — an AI search assistant for the Product Management team.
Your task is to analyze retrieved passages (from multiple sources) and determine whether they fully or partially answer the user’s query.

Rules:
1. Use only the supplied passages; never rely on outside knowledge or assumptions.
2. If the answer is partial or uncertain, explicitly note that — do not infer missing details.
3. Merge overlapping evidence from multiple passages into a single clear summary.
4. Quote or paraphrase 1–3 sentences of evidence for each cited passage.
5. Be concise, factual, and professional — no filler, disclaimers, or repetition.

Output Format (JSON only):
{
  "exists": boolean,                       // true if relevant info was found
  "answer": "short, synthesized summary (2–4 sentences max)",
  "citations": [
      {"url": string, "title": string, "evidence": "direct or paraphrased supporting text"}
  ]
}
Return strictly valid JSON. No markdown, no commentary, no explanations outside the JSON object.
""")

def build_user_payload(query: str, passages: list[dict], max_chars_per_passage: int = 900) -> str:
    # passages: [{"title":..., "url":..., "text":...}, ...]
    blocks = []
    for p in passages:
        snippet = (p.get("text") or p.get("excerpt") or "")[:max_chars_per_passage]
        blocks.append({
            "title": p.get("title",""),
            "url": p.get("url",""),
            "snippet": snippet,
            "source": p.get("source", "unknown")
        })
    return json.dumps({"query": query, "passages": blocks}, ensure_ascii=False)

def answer_with_citations(query: str, passages: list[dict]) -> dict:
    """Return dict: {exists, answer, citations: [{url,title,evidence}]}"""
    user_payload = build_user_payload(query, passages)

    prompt = (
        f"{SYSTEM_MSG}\n\n"
        "User Query and Passages (JSON):\n"
        f"{user_payload}\n\n"
        "Rules:\n"
        "- Cite only passages that directly support the answer.\n"
        "- Keep 'evidence' to 1–2 sentences copied from the snippet (no ellipses at both ends).\n"
        "- If unsure, set exists=false and explain briefly."
    )

    model = genai.GenerativeModel(MODEL_NAME)
    resp = model.generate_content(prompt)
    # robust parse
    text = resp.text.strip() if hasattr(resp, "text") else ""
    
    # Strip markdown code fences if present
    if text.startswith("```"):
        # Remove ```json or ``` at start and ``` at end
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    
    try:
        data = json.loads(text)
    except Exception:
        # fallback: wrap the raw text
        data = {"exists": False, "answer": text[:600], "citations": []}
    # minimal sanity
    data.setdefault("exists", False)
    data.setdefault("answer", "")
    data.setdefault("citations", [])
    return data


def answer_with_multiple_sources(query: str, qdrant_results: list[dict], slack_results: list[dict], confluence_results: list[dict]) -> dict:
    """
    Generate answer using multiple data sources (Qdrant, Slack, Confluence).
    
    Args:
        query: User query
        qdrant_results: Results from Qdrant vector search
        slack_results: Results from Slack search
        confluence_results: Results from Confluence search
    
    Returns:
        Dict with answer and citations from all sources
    """
    # Combine all results into a single passages list
    all_passages = []
    
    # Add Qdrant results
    for result in qdrant_results:
        all_passages.append({
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "text": result.get("text", ""),
            "source": "knowledge_base"
        })
    
    # Add Slack results
    for result in slack_results:
        all_passages.append({
            "title": f"Slack: #{result.get('channel', 'unknown')} - @{result.get('username', 'unknown')}",
            "url": result.get("permalink", ""),
            "text": result.get("text", ""),
            "source": "slack"
        })
    
    # Add Confluence results
    for result in confluence_results:
        all_passages.append({
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "text": result.get("excerpt", ""),
            "source": "confluence"
        })
    
    # Use the existing answer_with_citations function
    return answer_with_citations(query, all_passages)
