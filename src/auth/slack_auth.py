"""
Slack OAuth authentication for Streamlit.
Allows users to authenticate with Slack to access the application and their private channels.
"""

import streamlit as st
from slack_sdk.web import WebClient
from slack_sdk.errors import SlackApiError
import os
import secrets
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class SlackOAuthHandler:
    """Handles Slack OAuth authentication."""

    # OAuth scopes needed for the application
    USER_SCOPES = [
        'search:read',           # Search messages and files
        'channels:read',         # View basic channel info
        'channels:history',      # View messages in public channels
        'groups:read',          # View basic info about private channels user is in
        'groups:history',       # View messages in private channels user is in
        'users:read',           # View people in workspace
        'users:read.email',     # View email addresses of people
        'team:read',            # View workspace name, domain, etc
    ]

    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None, redirect_uri: Optional[str] = None):
        """
        Initialize Slack OAuth handler.

        Args:
            client_id: Slack OAuth client ID
            client_secret: Slack OAuth client secret
            redirect_uri: OAuth redirect URI (default: http://localhost:8501)
        """
        self.client_id = client_id or self._get_secret_or_env("SLACK_CLIENT_ID")
        self.client_secret = client_secret or self._get_secret_or_env("SLACK_CLIENT_SECRET")
        self.redirect_uri = redirect_uri or self._get_secret_or_env(
            "SLACK_REDIRECT_URI",
            "http://localhost:8501"
        )

        # Store whether OAuth is properly configured
        self.oauth_configured = bool(self.client_id and self.client_secret)

        # Initialize session state
        if "slack_auth_state" not in st.session_state:
            st.session_state.slack_auth_state = None
        if "slack_user_info" not in st.session_state:
            st.session_state.slack_user_info = None
        if "slack_token" not in st.session_state:
            st.session_state.slack_token = None
        if "slack_team_info" not in st.session_state:
            st.session_state.slack_team_info = None

    def _get_secret_or_env(self, name: str, default: str = "") -> str:
        """Get value from Streamlit secrets or environment variables."""
        try:
            if hasattr(st, 'secrets') and name in st.secrets:
                return st.secrets.get(name, default)
        except Exception:
            pass
        return os.getenv(name, default)

    def is_authenticated(self) -> bool:
        """Check if user is authenticated with Slack."""
        return st.session_state.get("slack_token") is not None

    def get_user_info(self) -> Optional[Dict[str, Any]]:
        """Get authenticated user information."""
        return st.session_state.get("slack_user_info")

    def get_team_info(self) -> Optional[Dict[str, Any]]:
        """Get Slack workspace/team information."""
        return st.session_state.get("slack_team_info")

    def generate_authorization_url(self) -> str:
        """Generate Slack OAuth authorization URL."""
        # Generate random state for CSRF protection
        state = secrets.token_urlsafe(32)
        st.session_state.slack_auth_state = state

        # Build authorization URL manually
        from urllib.parse import urlencode

        # Use comma-separated scopes for user_scope (matching Farah's implementation)
        user_scopes = ",".join(self.USER_SCOPES)

        params = {
            "client_id": self.client_id,
            "scope": "",  # Empty bot scope - required for user-only OAuth
            "user_scope": user_scopes,
            "redirect_uri": self.redirect_uri,
            "state": state
        }

        auth_url = f"https://slack.com/oauth/v2/authorize?{urlencode(params)}"

        return auth_url

    def handle_callback(self, code: str, state: str) -> bool:
        """
        Handle OAuth callback from Slack.

        Args:
            code: Authorization code from Slack
            state: State parameter for CSRF validation

        Returns:
            True if authentication successful, False otherwise
        """
        # Validate state for CSRF protection
        if state != st.session_state.get("slack_auth_state"):
            logger.error("OAuth state mismatch - possible CSRF attack")
            st.error("Authentication failed: Invalid state parameter")
            return False

        try:
            # Exchange code for token
            client = WebClient()
            response = client.oauth_v2_access(
                client_id=self.client_id,
                client_secret=self.client_secret,
                code=code,
                redirect_uri=self.redirect_uri
            )

            if not response.get("ok"):
                logger.error(f"OAuth token exchange failed: {response.get('error')}")
                st.error(f"Authentication failed: {response.get('error')}")
                return False

            # Get the user token (not bot token)
            authed_user = response.get("authed_user", {})
            access_token = authed_user.get("access_token")

            if not access_token:
                logger.error("No access token in OAuth response")
                st.error("Authentication failed: No access token received")
                return False

            # Store token
            st.session_state.slack_token = access_token

            # Get team/workspace info
            team_info = response.get("team", {})
            st.session_state.slack_team_info = {
                "id": team_info.get("id"),
                "name": team_info.get("name"),
            }

            # Get user info using the token
            user_client = WebClient(token=access_token)
            user_response = user_client.auth_test()

            if user_response.get("ok"):
                user_id = user_response.get("user_id")

                # Get detailed user profile
                user_profile_response = user_client.users_info(user=user_id)

                if user_profile_response.get("ok"):
                    user = user_profile_response.get("user", {})
                    profile = user.get("profile", {})

                    st.session_state.slack_user_info = {
                        "id": user.get("id"),
                        "name": user.get("name"),
                        "real_name": user.get("real_name") or profile.get("real_name"),
                        "email": profile.get("email"),
                        "display_name": profile.get("display_name"),
                        "image": profile.get("image_192") or profile.get("image_72"),
                        "team_id": user.get("team_id"),
                    }

            logger.info(f"User authenticated: {st.session_state.slack_user_info.get('email')}")
            return True

        except SlackApiError as e:
            logger.error(f"Slack API error during OAuth: {e}")
            st.error(f"Authentication failed: {e.response.get('error', str(e))}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during OAuth: {e}")
            st.error(f"Authentication failed: {str(e)}")
            return False

    def logout(self) -> None:
        """Log out user and clear session."""
        # Revoke token if possible
        if st.session_state.get("slack_token"):
            try:
                client = WebClient(token=st.session_state.slack_token)
                client.auth_revoke()
            except Exception as e:
                logger.warning(f"Failed to revoke token: {e}")

        # Clear session state
        st.session_state.slack_auth_state = None
        st.session_state.slack_user_info = None
        st.session_state.slack_token = None
        st.session_state.slack_team_info = None

        # Clear any cached Slack clients
        keys_to_remove = [k for k in st.session_state.keys() if k.startswith('slack_client_')]
        for key in keys_to_remove:
            del st.session_state[key]

        logger.info("User logged out")

    def require_auth(self) -> bool:
        """
        Require authentication before proceeding.
        Shows login UI if not authenticated.

        Returns:
            True if authenticated, False if showing login UI
        """
        # Check if OAuth is configured
        if not self.oauth_configured:
            self._show_setup_guide()
            return False

        # Check for OAuth callback parameters in URL
        query_params = st.query_params

        if "code" in query_params and "state" in query_params:
            code = query_params["code"]
            state = query_params["state"]

            # Handle callback
            if self.handle_callback(code, state):
                st.success("Successfully authenticated with Slack!")
                # Clear query parameters
                st.query_params.clear()
                st.rerun()
            else:
                st.query_params.clear()
                return False

        # Check if already authenticated
        if self.is_authenticated():
            return True

        # Show login UI
        self._show_login_ui()
        return False

    def _show_login_ui(self) -> None:
        """Display Slack login UI using Streamlit native components."""
        # Simple CSS for button styling only
        st.markdown("""
            <style>
            .stApp {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }

            div[data-testid="stVerticalBlock"] > div:has(div.element-container) {
                background-color: white;
                padding: 3rem 2rem;
                border-radius: 16px;
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
            }
            </style>
        """, unsafe_allow_html=True)

        # Add spacing
        st.write("")
        st.write("")
        st.write("")

        # Create centered layout
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            # Logo
            st.markdown("<div style='text-align: center; font-size: 4rem; margin-bottom: 1rem;'>💬</div>", unsafe_allow_html=True)

            # Title
            st.markdown("<h1 style='text-align: center; color: #1a202c; margin-bottom: 0.5rem;'>Internal PM Chat</h1>", unsafe_allow_html=True)

            # Subtitle
            st.markdown("<p style='text-align: center; color: #718096; margin-bottom: 2rem;'>Your unified workspace for Slack conversations, Confluence docs, and team knowledge</p>", unsafe_allow_html=True)

            # Generate auth URL
            auth_url = self.generate_authorization_url()

            # Sign in button with better styling
            st.markdown(f"""
                <div style='text-align: center; margin: 2rem 0;'>
                    <a href="{auth_url}"
                       style='display: inline-block;
                              background: #4A154B;
                              color: white;
                              padding: 12px 32px;
                              border-radius: 8px;
                              text-decoration: none;
                              font-weight: 600;
                              font-size: 1rem;
                              transition: background 0.3s;'>
                        Sign in with Slack
                    </a>
                </div>
            """, unsafe_allow_html=True)

            st.write("")

            # Divider
            st.markdown("---")

            # Features
            st.markdown("### Why Sign in with Slack?")

            col_a, col_b = st.columns([1, 9])
            with col_a:
                st.write("🔒")
            with col_b:
                st.markdown("**Private Channel Access**  \nSearch your private channels and conversations securely")

            col_a, col_b = st.columns([1, 9])
            with col_a:
                st.write("🎯")
            with col_b:
                st.markdown("**Personalized Results**  \nGet search results tailored to your permissions")

            col_a, col_b = st.columns([1, 9])
            with col_a:
                st.write("⚡")
            with col_b:
                st.markdown("**Unified Search**  \nSearch across Slack, Confluence, and documentation")

            st.write("")
            st.caption("🔐 Secured by Slack OAuth 2.0")

    def _show_setup_guide(self) -> None:
        """Display setup guide when OAuth credentials are not configured."""
        st.error("⚙️ **Slack OAuth Not Configured**")

        st.markdown("""
        ## Setup Required

        The application needs Slack OAuth credentials to function. Please follow these steps:

        ### 1. Create a Slack App

        1. Go to [https://api.slack.com/apps](https://api.slack.com/apps)
        2. Click **"Create New App"** → **"From scratch"**
        3. Enter App Name: `Internal PM Chat`
        4. Select your workspace
        5. Click **"Create App"**

        ### 2. Configure OAuth & Permissions

        1. Go to **"OAuth & Permissions"** in sidebar
        2. Add Redirect URL:
           - Local: `http://localhost:8501`
           - Production: Your deployment URL
        3. Add these **User Token Scopes**:
           - `search:read`
           - `channels:read`
           - `channels:history`
           - `groups:read`
           - `groups:history`
           - `users:read`
           - `users:read.email`
           - `team:read`

        ### 3. Get Credentials

        1. Go to **"Basic Information"**
        2. Copy **Client ID** and **Client Secret**

        ### 4. Configure Secrets

        Add to `.streamlit/secrets.toml`:

        ```toml
        SLACK_CLIENT_ID = "your_client_id_here"
        SLACK_CLIENT_SECRET = "your_client_secret_here"
        SLACK_REDIRECT_URI = "http://localhost:8501"
        ```

        Or set environment variables:
        ```bash
        export SLACK_CLIENT_ID="your_client_id_here"
        export SLACK_CLIENT_SECRET="your_client_secret_here"
        ```

        ### 5. Restart the Application

        After adding credentials, restart Streamlit to apply changes.

        ---

        **Need help?** Check the documentation or contact your administrator.
        """)


def get_slack_auth_handler() -> SlackOAuthHandler:
    """Get or create SlackOAuthHandler instance."""
    if "slack_auth_handler" not in st.session_state:
        st.session_state.slack_auth_handler = SlackOAuthHandler()
    return st.session_state.slack_auth_handler


__all__ = ["SlackOAuthHandler", "get_slack_auth_handler"]
