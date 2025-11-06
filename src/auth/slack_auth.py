"""
Slack OAuth authentication for Streamlit.
Allows users to authenticate with Slack to access the application and their private channels.
"""

import streamlit as st
from slack_sdk.web import WebClient
from slack_sdk.oauth import AuthorizeUrlGenerator, RedirectUriPageRenderer
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

        if not self.client_id or not self.client_secret:
            raise ValueError(
                "Missing Slack OAuth credentials. Set SLACK_CLIENT_ID and SLACK_CLIENT_SECRET "
                "in environment variables or Streamlit secrets."
            )

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

        # Generate authorization URL
        authorize_url_generator = AuthorizeUrlGenerator(
            client_id=self.client_id,
            scopes=self.USER_SCOPES,
            user_scopes=self.USER_SCOPES
        )

        auth_url = authorize_url_generator.generate(
            state=state,
            redirect_uri=self.redirect_uri
        )

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
        """Display Slack login UI."""
        st.markdown("""
            <style>
            .slack-login-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 60vh;
                text-align: center;
            }
            .slack-logo {
                font-size: 4rem;
                margin-bottom: 2rem;
            }
            .login-title {
                font-size: 2.5rem;
                font-weight: 700;
                color: #1f77b4;
                margin-bottom: 1rem;
            }
            .login-subtitle {
                font-size: 1.2rem;
                color: #666;
                margin-bottom: 3rem;
            }
            .slack-button {
                display: inline-block;
                background: #4A154B;
                color: white;
                padding: 12px 24px;
                border-radius: 4px;
                text-decoration: none;
                font-weight: 600;
                font-size: 1.1rem;
                transition: background 0.3s;
            }
            .slack-button:hover {
                background: #611f69;
                text-decoration: none;
                color: white;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="slack-login-container">
                <div class="slack-logo">💬</div>
                <div class="login-title">Internal PM Chat</div>
                <div class="login-subtitle">Sign in with Slack to access your workspace</div>
            </div>
        """, unsafe_allow_html=True)

        # Generate auth URL
        auth_url = self.generate_authorization_url()

        # Create centered button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.markdown(f"""
                <a href="{auth_url}" class="slack-button" target="_self">
                    Sign in with Slack
                </a>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.info("""
            **Why Slack authentication?**
            - Access your private channels and conversations
            - Personalized search results based on your permissions
            - Secure access with your Slack workspace credentials
        """)


def get_slack_auth_handler() -> SlackOAuthHandler:
    """Get or create SlackOAuthHandler instance."""
    if "slack_auth_handler" not in st.session_state:
        st.session_state.slack_auth_handler = SlackOAuthHandler()
    return st.session_state.slack_auth_handler


__all__ = ["SlackOAuthHandler", "get_slack_auth_handler"]
