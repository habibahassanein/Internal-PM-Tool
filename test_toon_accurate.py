"""
Test script with accurate token counting using tiktoken.
"""

import json
import toon
import tiktoken
from src.handler.gemini_handler import build_user_payload

# Use cl100k_base encoding (used by GPT-3.5/4, similar to Gemini)
encoding = tiktoken.get_encoding("cl100k_base")

# Sample test data simulating real PM tool passages
sample_passages = [
    {
        "title": "Feature Release Timeline Q4 2024",
        "url": "https://confluence.example.com/feature-release-q4",
        "text": "The new authentication system is scheduled for release in late Q4 2024. The engineering team has completed 80% of the implementation. QA testing will begin on November 15th. Product Manager Jane Smith confirmed the feature will include SSO integration, multi-factor authentication, and session management improvements.",
        "source": "confluence"
    },
    {
        "title": "Customer Support Ticket #5421",
        "url": "https://zendesk.example.com/tickets/5421",
        "text": "Customer reported slow loading times on the dashboard page. Issue occurs when viewing projects with more than 100 tasks. Browser console shows multiple API calls timing out after 30 seconds. Customer is on Enterprise plan. Priority: High. Status: In Progress.",
        "source": "zendesk"
    },
    {
        "title": "JIRA-2891: Optimize Dashboard Performance",
        "url": "https://jira.example.com/browse/JIRA-2891",
        "text": "Epic for dashboard performance optimization. Includes tickets for reducing API calls, implementing lazy loading, and adding caching layer. Current sprint focus. Assigned to Backend Team. Story points: 21. Sprint 47. Linked to 3 customer escalations.",
        "source": "jira"
    },
    {
        "title": "Slack: #product-releases discussion",
        "url": "https://slack.example.com/archives/C12345/p1698765432",
        "text": "PM Update from Sarah: Authentication feature is on track for November 30th release. We'll need to coordinate with marketing for the announcement. Engineering confirmed all security reviews are complete. Documentation team is updating the user guides.",
        "source": "slack"
    },
    {
        "title": "Knowledge Base: Dashboard Best Practices",
        "url": "https://kb.example.com/dashboard-best-practices",
        "text": "For optimal dashboard performance, limit the number of visible tasks to 50 per page. Use filters to narrow down results. Enable browser caching in settings. Contact support if loading times exceed 5 seconds. Enterprise customers can request dedicated database instances.",
        "source": "knowledge_base"
    }
]

def count_tokens_accurate(text: str) -> int:
    """Accurate token count using tiktoken."""
    return len(encoding.encode(text))

def test_toon_vs_json_accurate():
    """Compare TOON vs JSON encoding with accurate token counting."""

    query = "What is the status of the authentication feature release and are there any performance issues?"

    print("=" * 80)
    print("🎯 TOON Integration Test - ACCURATE Token Analysis")
    print("=" * 80)
    print()

    # Test with JSON (traditional)
    print("📊 JSON Format (Traditional)...")
    json_payload = build_user_payload(query, sample_passages, use_toon=False)
    json_tokens = count_tokens_accurate(json_payload)

    print(f"   Tokens: {json_tokens:,}")
    print(f"   Characters: {len(json_payload):,}")
    print()

    # Test with TOON
    print("🎒 TOON Format (Optimized)...")
    toon_payload = build_user_payload(query, sample_passages, use_toon=True)
    toon_tokens = count_tokens_accurate(toon_payload)

    print(f"   Tokens: {toon_tokens:,}")
    print(f"   Characters: {len(toon_payload):,}")
    print()

    # Calculate savings
    token_savings = json_tokens - toon_tokens
    token_savings_pct = (token_savings / json_tokens) * 100

    print("=" * 80)
    print("💰 TOKEN SAVINGS")
    print("=" * 80)
    print(f"JSON tokens:  {json_tokens:,}")
    print(f"TOON tokens:  {toon_tokens:,}")
    print(f"Saved:        {token_savings:,} tokens ({token_savings_pct:.1f}% reduction)")
    print()

    # Show actual output comparison
    print("=" * 80)
    print("📝 OUTPUT COMPARISON")
    print("=" * 80)
    print()
    print("JSON (first 400 chars):")
    print("-" * 80)
    print(json_payload[:400])
    print("...")
    print()
    print("TOON (first 400 chars):")
    print("-" * 80)
    print(toon_payload[:400])
    print("...")
    print()

    # Extrapolate to monthly savings
    queries_per_day = 100
    days_per_month = 30

    monthly_json_tokens = json_tokens * queries_per_day * days_per_month
    monthly_toon_tokens = toon_tokens * queries_per_day * days_per_month
    monthly_token_savings = monthly_json_tokens - monthly_toon_tokens

    # Gemini pricing (approximate for input tokens)
    cost_per_million_tokens = 0.075  # USD
    monthly_cost_json = (monthly_json_tokens / 1_000_000) * cost_per_million_tokens
    monthly_cost_toon = (monthly_toon_tokens / 1_000_000) * cost_per_million_tokens
    monthly_cost_savings = monthly_cost_json - monthly_cost_toon

    print("=" * 80)
    print("📈 PROJECTED SAVINGS (100 queries/day)")
    print("=" * 80)
    print(f"Monthly tokens saved:     {monthly_token_savings:,}")
    print(f"Annual tokens saved:      {monthly_token_savings * 12:,}")
    print()
    print(f"Monthly cost savings:     ${monthly_cost_savings:.2f}")
    print(f"Annual cost savings:      ${monthly_cost_savings * 12:.2f}")
    print()
    print("💡 Note: Savings apply to passage data only. Total savings increase")
    print("   when factoring in system messages and full prompt structure.")
    print()

    print("=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)
    print()
    print(f"✨ TOON achieves {token_savings_pct:.1f}% token reduction on passage data")
    print(f"💰 Estimated monthly savings: ${monthly_cost_savings:.2f}")
    print()

    return {
        "json_tokens": json_tokens,
        "toon_tokens": toon_tokens,
        "savings_pct": token_savings_pct,
        "monthly_savings_tokens": monthly_token_savings,
        "monthly_savings_usd": monthly_cost_savings
    }

if __name__ == "__main__":
    try:
        results = test_toon_vs_json_accurate()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
