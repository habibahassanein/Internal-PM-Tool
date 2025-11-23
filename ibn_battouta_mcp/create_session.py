#!/usr/bin/env python3
"""
Manual session creation script.
Use this to create a session ID when OAuth flow is not accessible.

Usage:
    python create_session.py <slack_user_token>

Example:
    python create_session.py xoxp-your-slack-token-here
"""

import sys
from auth.session_manager import get_session_manager
from slack_sdk.web import WebClient


def create_session_manually(slack_token: str) -> str:
    """
    Create a session manually with a Slack user token.

    Args:
        slack_token: Slack user OAuth token (starts with xoxp-)

    Returns:
        session_id: The created session ID
    """
    try:
        # Validate token and get user info
        client = WebClient(token=slack_token)
        user_info = client.users_identity()
        user_name = user_info["user"]["name"]
        user_email = user_info["user"].get("email", "")

        print(f"✓ Token validated for user: {user_name} ({user_email})")

        # Create session
        session_mgr = get_session_manager()
        session_id = session_mgr.create_session(
            slack_token=slack_token,
            user_name=user_name,
            user_email=user_email
        )

        print(f"\n{'='*70}")
        print(f"✓ Session created successfully!")
        print(f"{'='*70}")
        print(f"\nYour Session ID:")
        print(f"  {session_id}")
        print(f"\nAdd this to your Claude Desktop config:")
        print(f"  ~/Library/Application Support/Claude/claude_desktop_config.json")
        print(f"\n{{\n  \"mcpServers\": {{\n    \"ibn-battouta\": {{")
        print(f"      \"url\": \"http://localhost:8081/mcp\",")
        print(f"      \"transport\": \"http\",")
        print(f"      \"headers\": {{")
        print(f"        \"session-id\": \"{session_id}\"")
        print(f"      }}")
        print(f"    }}")
        print(f"  }}")
        print(f"}}")
        print(f"\nSession expires after 30 days of inactivity.")
        print(f"{'='*70}\n")

        return session_id

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        sys.exit(1)


def print_usage():
    """Print usage instructions."""
    print("\n" + "="*70)
    print("Manual Session Creation Script")
    print("="*70)
    print("\nThis script creates a session ID using your Slack user token.")
    print("\nSteps to get your Slack user token:")
    print("  1. Go to https://api.slack.com/apps")
    print("  2. Select your app (Ibn Battouta / Client ID: 122067183344.9737634042256)")
    print("  3. Go to 'OAuth & Permissions' in the sidebar")
    print("  4. Under 'User Token Scopes', ensure these scopes are added:")
    print("     - channels:history")
    print("     - channels:read")
    print("     - groups:history")
    print("     - groups:read")
    print("     - search:read")
    print("     - users:read")
    print("  5. Click 'Install to Workspace' (or 'Reinstall to Workspace')")
    print("  6. Authorize the app")
    print("  7. Copy the 'User OAuth Token' (starts with xoxp-)")
    print("\nUsage:")
    print("  python create_session.py <slack_user_token>")
    print("\nExample:")
    print("  python create_session.py xoxp-1234567890-1234567890-abcdefghij")
    print("="*70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print_usage()
        sys.exit(1)

    slack_token = sys.argv[1]

    if not slack_token.startswith("xoxp-"):
        print("\n✗ Error: Token must be a Slack user token (starts with 'xoxp-')")
        print("  You provided a token starting with:", slack_token[:10])
        print_usage()
        sys.exit(1)

    create_session_manually(slack_token)
