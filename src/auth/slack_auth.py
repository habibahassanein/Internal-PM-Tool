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

        scopes = ",".join(self.USER_SCOPES)

        params = {
            "client_id": self.client_id,
            "scope": scopes,
            "user_scope": scopes,
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
        """Display Slack login UI."""
        # Hide Streamlit's default header and footer for cleaner look
        st.markdown("""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

            /* Hide Streamlit branding */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}

            /* Reset body styles */
            .stApp {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }

            .login-card {
                margin: 0 auto;
                background: white;
                border-radius: 24px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                padding: 3rem 2.5rem;
                max-width: 480px;
                width: 100%;
                text-align: center;
                animation: fadeInUp 0.6s ease-out;
            }

            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(30px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .slack-logo-wrapper {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 80px;
                height: 80px;
                background: linear-gradient(135deg, #4A154B 0%, #611f69 100%);
                border-radius: 20px;
                margin-bottom: 1.5rem;
                box-shadow: 0 8px 16px rgba(74, 21, 75, 0.3);
                animation: bounceIn 0.8s ease-out 0.2s backwards;
            }

            @keyframes bounceIn {
                0% {
                    opacity: 0;
                    transform: scale(0.3);
                }
                50% {
                    transform: scale(1.05);
                }
                100% {
                    opacity: 1;
                    transform: scale(1);
                }
            }

            .slack-logo {
                font-size: 3rem;
                filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
            }

            .login-title {
                font-size: 2rem;
                font-weight: 700;
                color: #1a202c;
                margin-bottom: 0.75rem;
                letter-spacing: -0.5px;
                animation: fadeIn 0.6s ease-out 0.3s backwards;
            }

            @keyframes fadeIn {
                from {
                    opacity: 0;
                }
                to {
                    opacity: 1;
                }
            }

            .login-subtitle {
                font-size: 1.05rem;
                color: #718096;
                margin-bottom: 2.5rem;
                line-height: 1.6;
                animation: fadeIn 0.6s ease-out 0.4s backwards;
            }

            .slack-button {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                gap: 12px;
                background: linear-gradient(135deg, #4A154B 0%, #611f69 100%);
                color: white;
                padding: 16px 32px;
                border-radius: 12px;
                text-decoration: none;
                font-weight: 600;
                font-size: 1.05rem;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                box-shadow: 0 4px 12px rgba(74, 21, 75, 0.4);
                width: 100%;
                max-width: 280px;
                animation: fadeIn 0.6s ease-out 0.5s backwards;
            }

            .slack-button:hover {
                background: linear-gradient(135deg, #611f69 0%, #7e2b80 100%);
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(74, 21, 75, 0.5);
                text-decoration: none;
                color: white;
            }

            .slack-button:active {
                transform: translateY(0);
            }

            .slack-button svg {
                width: 24px;
                height: 24px;
            }

            .features-grid {
                display: grid;
                grid-template-columns: 1fr;
                gap: 1rem;
                margin-top: 2.5rem;
                padding-top: 2.5rem;
                border-top: 1px solid #e2e8f0;
                animation: fadeIn 0.6s ease-out 0.6s backwards;
            }

            .feature-item {
                display: flex;
                align-items: flex-start;
                text-align: left;
                gap: 12px;
                padding: 0.5rem;
            }

            .feature-icon {
                flex-shrink: 0;
                width: 32px;
                height: 32px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.1rem;
            }

            .feature-text {
                flex: 1;
            }

            .feature-title {
                font-weight: 600;
                color: #2d3748;
                font-size: 0.95rem;
                margin-bottom: 0.25rem;
            }

            .feature-desc {
                font-size: 0.85rem;
                color: #718096;
                line-height: 1.5;
            }

            .security-badge {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                background: #f7fafc;
                color: #4a5568;
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 0.85rem;
                margin-top: 1.5rem;
                animation: fadeIn 0.6s ease-out 0.7s backwards;
            }

            .security-badge svg {
                width: 16px;
                height: 16px;
            }
            </style>
        """, unsafe_allow_html=True)

        # Add spacing at top
        st.markdown("<br><br>", unsafe_allow_html=True)

        # Generate auth URL
        auth_url = self.generate_authorization_url()

        # Create centered layout using columns
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.markdown(f"""
                <div class="login-card">
                    <div class="slack-logo-wrapper">
                        <div class="slack-logo">💬</div>
                    </div>

                    <h1 class="login-title">Internal PM Chat</h1>
                    <p class="login-subtitle">
                        Your unified workspace for Slack conversations, Confluence docs,
                        and team knowledge - all in one intelligent search
                    </p>

                    <a href="{auth_url}" class="slack-button" target="_self">
                        <svg viewBox="0 0 24 24" fill="currentColor">
                            <path d="M5.042 15.165a2.528 2.528 0 0 1-2.52 2.523A2.528 2.528 0 0 1 0 15.165a2.527 2.527 0 0 1 2.522-2.52h2.52v2.52zM6.313 15.165a2.527 2.527 0 0 1 2.521-2.52 2.527 2.527 0 0 1 2.521 2.52v6.313A2.528 2.528 0 0 1 8.834 24a2.528 2.528 0 0 1-2.521-2.522v-6.313zM8.834 5.042a2.528 2.528 0 0 1-2.521-2.52A2.528 2.528 0 0 1 8.834 0a2.528 2.528 0 0 1 2.521 2.522v2.52H8.834zM8.834 6.313a2.528 2.528 0 0 1 2.521 2.521 2.528 2.528 0 0 1-2.521 2.521H2.522A2.528 2.528 0 0 1 0 8.834a2.528 2.528 0 0 1 2.522-2.521h6.312zM18.956 8.834a2.528 2.528 0 0 1 2.522-2.521A2.528 2.528 0 0 1 24 8.834a2.528 2.528 0 0 1-2.522 2.521h-2.522V8.834zM17.688 8.834a2.528 2.528 0 0 1-2.523 2.521 2.527 2.527 0 0 1-2.52-2.521V2.522A2.527 2.527 0 0 1 15.165 0a2.528 2.528 0 0 1 2.523 2.522v6.312zM15.165 18.956a2.528 2.528 0 0 1 2.523 2.522A2.528 2.528 0 0 1 15.165 24a2.527 2.527 0 0 1-2.52-2.522v-2.522h2.52zM15.165 17.688a2.527 2.527 0 0 1-2.52-2.523 2.526 2.526 0 0 1 2.52-2.52h6.313A2.527 2.527 0 0 1 24 15.165a2.528 2.528 0 0 1-2.522 2.523h-6.313z"/>
                        </svg>
                        Sign in with Slack
                    </a>

                    <div class="security-badge">
                        <svg viewBox="0 0 24 24" fill="currentColor">
                            <path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm0 10.99h7c-.53 4.12-3.28 7.79-7 8.94V12H5V6.3l7-3.11v8.8z"/>
                        </svg>
                        Secured by Slack OAuth 2.0
                    </div>

                    <div class="features-grid">
                        <div class="feature-item">
                            <div class="feature-icon">🔒</div>
                            <div class="feature-text">
                                <div class="feature-title">Private Channel Access</div>
                                <div class="feature-desc">Search your private channels and conversations securely</div>
                            </div>
                        </div>

                        <div class="feature-item">
                            <div class="feature-icon">🎯</div>
                            <div class="feature-text">
                                <div class="feature-title">Personalized Results</div>
                                <div class="feature-desc">Get search results tailored to your permissions</div>
                            </div>
                        </div>

                        <div class="feature-item">
                            <div class="feature-icon">⚡</div>
                            <div class="feature-text">
                                <div class="feature-title">Unified Search</div>
                                <div class="feature-desc">Search across Slack, Confluence, and documentation</div>
                            </div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

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
