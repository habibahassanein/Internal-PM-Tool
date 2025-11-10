"""
Test script to verify TOON integration and measure token savings.
"""

import json
import toon
from src.handler.gemini_handler import build_user_payload

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

def count_tokens_estimate(text: str) -> int:
    """
    Rough token count estimate (1 token ≈ 4 characters for English text).
    More accurate with tiktoken, but this gives a ballpark figure.
    """
    return len(text) // 4

def test_toon_vs_json():
    """Compare TOON vs JSON encoding for token efficiency."""

    query = "What is the status of the authentication feature release and are there any performance issues?"

    print("=" * 80)
    print("TOON Integration Test - Token Savings Analysis")
    print("=" * 80)
    print()

    # Test with JSON (traditional)
    print("📊 Testing JSON Format (Traditional)...")
    json_payload = build_user_payload(query, sample_passages, use_toon=False)
    json_length = len(json_payload)
    json_tokens = count_tokens_estimate(json_payload)

    print(f"   Length: {json_length:,} characters")
    print(f"   Estimated tokens: {json_tokens:,}")
    print()
    print("   Sample output (first 500 chars):")
    print("   " + "-" * 76)
    print("   " + json_payload[:500].replace("\n", "\n   "))
    print("   ...")
    print()

    # Test with TOON
    print("🎒 Testing TOON Format (Optimized)...")
    toon_payload = build_user_payload(query, sample_passages, use_toon=True)
    toon_length = len(toon_payload)
    toon_tokens = count_tokens_estimate(toon_payload)

    print(f"   Length: {toon_length:,} characters")
    print(f"   Estimated tokens: {toon_tokens:,}")
    print()
    print("   Sample output (first 500 chars):")
    print("   " + "-" * 76)
    print("   " + toon_payload[:500].replace("\n", "\n   "))
    print("   ...")
    print()

    # Calculate savings
    char_savings = json_length - toon_length
    char_savings_pct = (char_savings / json_length) * 100
    token_savings = json_tokens - toon_tokens
    token_savings_pct = (token_savings / json_tokens) * 100

    print("=" * 80)
    print("💰 SAVINGS ANALYSIS")
    print("=" * 80)
    print(f"Character reduction: {char_savings:,} ({char_savings_pct:.1f}%)")
    print(f"Token reduction: {token_savings:,} ({token_savings_pct:.1f}%)")
    print()

    # Extrapolate to monthly savings
    queries_per_day = 100
    days_per_month = 30

    monthly_json_tokens = json_tokens * queries_per_day * days_per_month
    monthly_toon_tokens = toon_tokens * queries_per_day * days_per_month
    monthly_token_savings = monthly_json_tokens - monthly_toon_tokens

    # Gemini pricing (approximate)
    cost_per_million_tokens = 0.075  # USD
    monthly_cost_json = (monthly_json_tokens / 1_000_000) * cost_per_million_tokens
    monthly_cost_toon = (monthly_toon_tokens / 1_000_000) * cost_per_million_tokens
    monthly_cost_savings = monthly_cost_json - monthly_cost_toon

    print(f"📈 Monthly Projections ({queries_per_day} queries/day):")
    print(f"   JSON tokens/month: {monthly_json_tokens:,}")
    print(f"   TOON tokens/month: {monthly_toon_tokens:,}")
    print(f"   Token savings/month: {monthly_token_savings:,}")
    print()
    print(f"💵 Cost Estimates (Gemini API @ ${cost_per_million_tokens}/1M tokens):")
    print(f"   JSON cost/month: ${monthly_cost_json:.2f}")
    print(f"   TOON cost/month: ${monthly_cost_toon:.2f}")
    print(f"   💰 Monthly savings: ${monthly_cost_savings:.2f}")
    print(f"   💰 Annual savings: ${monthly_cost_savings * 12:.2f}")
    print()

    print("=" * 80)
    print("✅ TOON Integration Test Complete!")
    print("=" * 80)

    return {
        "json_tokens": json_tokens,
        "toon_tokens": toon_tokens,
        "savings_pct": token_savings_pct,
        "monthly_savings_usd": monthly_cost_savings
    }

if __name__ == "__main__":
    try:
        results = test_toon_vs_json()
        print()
        print(f"✨ Result: TOON reduces tokens by {results['savings_pct']:.1f}%")
        print(f"💰 Potential savings: ${results['monthly_savings_usd']:.2f}/month")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
