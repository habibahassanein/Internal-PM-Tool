"""
Google OAuth authentication for Streamlit with domain restriction.
Only allows users with valid Incorta domain emails to access the application.
"""

import streamlit as st
from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import os
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import secrets


class GoogleOAuthHandler:
    """Handles Google OAuth authentication with domain restrictions."""

    # Allowed email domains
    ALLOWED_DOMAINS = ["incorta.com"]

    # OAuth scopes
    SCOPES = [
        'openid',
        'https://www.googleapis.com/auth/userinfo.email',
        'https://www.googleapis.com/auth/userinfo.profile'
    ]

    def __init__(self, client_secrets_file: Optional[str] = None, redirect_uri: Optional[str] = None):
        """
        Initialize OAuth handler.

        Args:
            client_secrets_file: Path to Google OAuth client secrets JSON
            redirect_uri: OAuth redirect URI (default: http://localhost:8501)
        """
        self.client_secrets_file = client_secrets_file or os.getenv(
            "GOOGLE_CLIENT_SECRETS_FILE",
            "client_secrets.json"
        )
        self.redirect_uri = redirect_uri or os.getenv(
            "OAUTH_REDIRECT_URI",
            "http://localhost:8501"
        )

        # Initialize session state
        if "auth_state" not in st.session_state:
            st.session_state.auth_state = None
        if "user_info" not in st.session_state:
            st.session_state.user_info = None
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False

    def _create_flow(self) -> Flow:
        """Create OAuth flow object."""
        try:
            # Try to load from file first (local development)
            flow = Flow.from_client_secrets_file(
                self.client_secrets_file,
                scopes=self.SCOPES,
                redirect_uri=self.redirect_uri
            )
            return flow
        except FileNotFoundError:
            # Fall back to Streamlit secrets (cloud deployment)
            try:
                if "web" in st.secrets:
                    client_config = {
                        "web": {
                            "client_id": st.secrets["web"]["client_id"],
                            "client_secret": st.secrets["web"]["client_secret"],
                            "auth_uri": st.secrets["web"].get("auth_uri", "https://accounts.google.com/o/oauth2/auth"),
                            "token_uri": st.secrets["web"].get("token_uri", "https://oauth2.googleapis.com/token"),
                            "auth_provider_x509_cert_url": st.secrets["web"].get("auth_provider_x509_cert_url", "https://www.googleapis.com/oauth2/v1/certs"),
                            "redirect_uris": [self.redirect_uri]
                        }
                    }
                    flow = Flow.from_client_config(
                        client_config,
                        scopes=self.SCOPES,
                        redirect_uri=self.redirect_uri
                    )
                    return flow
                else:
                    raise KeyError("OAuth configuration not found in secrets")
            except (KeyError, AttributeError) as e:
                st.error(f"❌ OAuth configuration not found")
                st.info("""
                **For local development**: Place `client_secrets.json` in the project root

                **For Streamlit Cloud**: Add OAuth credentials to your app secrets

                See QUICK_START_OAUTH.md or STREAMLIT_CLOUD_SETUP.md for instructions.
                """)
                st.stop()

    def _validate_domain(self, email: str) -> bool:
        """
        Validate if email belongs to allowed domain.

        Args:
            email: User's email address

        Returns:
            True if domain is allowed, False otherwise
        """
        if not email:
            return False

        domain = email.split('@')[-1].lower()
        return domain in self.ALLOWED_DOMAINS

    def get_authorization_url(self) -> tuple[str, str]:
        """
        Generate OAuth authorization URL.

        Returns:
            Tuple of (authorization_url, state)
        """
        flow = self._create_flow()

        # Generate state token for CSRF protection
        state = secrets.token_urlsafe(32)

        authorization_url, _ = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            state=state,
            prompt='select_account'
        )

        return authorization_url, state

    def handle_oauth_callback(self, code: str, state: str) -> Dict[str, Any]:
        """
        Handle OAuth callback and exchange code for tokens.

        Args:
            code: Authorization code from OAuth callback
            state: State token for CSRF validation

        Returns:
            User info dictionary
        """
        # Note: CSRF validation is difficult with Streamlit due to session state limitations
        # The state parameter is still passed for Google's validation
        # In production, consider using a database to store state tokens

        flow = self._create_flow()

        # Exchange authorization code for tokens
        flow.fetch_token(code=code)
        credentials = flow.credentials

        # Verify ID token and get user info
        request = google_requests.Request()
        id_info = id_token.verify_oauth2_token(
            credentials.id_token,
            request,
            flow.client_config['client_id']
        )

        # Extract user information
        user_info = {
            'email': id_info.get('email'),
            'name': id_info.get('name'),
            'picture': id_info.get('picture'),
            'email_verified': id_info.get('email_verified', False),
            'sub': id_info.get('sub'),  # Unique user ID
            'authenticated_at': datetime.utcnow().isoformat()
        }

        # Validate domain
        if not self._validate_domain(user_info['email']):
            raise ValueError(
                f"Access denied. Only {', '.join(self.ALLOWED_DOMAINS)} email addresses are allowed."
            )

        # Validate email is verified
        if not user_info['email_verified']:
            raise ValueError("Email address not verified. Please verify your email with Google.")

        return user_info

    def login(self):
        """Display login interface and handle authentication flow."""

        # Check if already authenticated in session
        if st.session_state.get("authenticated"):
            return True

        # Try to restore authentication from database session
        if "auth_session_id" in st.session_state:
            from .session_manager import get_session_manager
            session_mgr = get_session_manager()
            if session_mgr.is_session_valid(st.session_state.auth_session_id):
                # Session is still valid, restore authentication
                st.session_state.authenticated = True
                return True

        # Check for OAuth callback parameters
        query_params = st.query_params

        if "code" in query_params and "state" in query_params:
            # Handle OAuth callback
            try:
                code = query_params["code"]
                state = query_params["state"]

                user_info = self.handle_oauth_callback(code, state)

                # Store user info in session
                st.session_state.user_info = user_info
                st.session_state.authenticated = True

                # Mark that we just authenticated
                st.session_state.just_authenticated = True

                # Clear query parameters
                st.query_params.clear()

                st.success(f"✅ Successfully authenticated as {user_info['name']} ({user_info['email']})")

                # Force a clean reload
                st.rerun()

            except Exception as e:
                st.error(f"❌ Authentication failed: {str(e)}")
                st.session_state.authenticated = False
                st.session_state.user_info = None

                # Clear query parameters
                st.query_params.clear()

                # Show login button again
                if st.button("Try Again"):
                    st.rerun()
                st.stop()

        else:
            # Show login page
            self._show_login_page()
            st.stop()

    def _show_login_page(self):
        """Display the login page."""
        st.title("🔐 Internal PM Tool - Authentication Required")

        st.markdown("""
        ### Welcome to the Internal PM Chat Assistant

        This application provides access to sensitive internal data including:
        - Confluence documentation
        - Slack conversations
        - Zendesk tickets
        - Jira issues

        **Access is restricted to Incorta employees only.**

        Please sign in with your **@incorta.com** Google account to continue.
        """)

        # Generate authorization URL
        try:
            auth_url, state = self.get_authorization_url()

            # Note: State is included in the OAuth URL but cannot be reliably validated
            # in Streamlit due to session state reset during redirects

            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                # Use a link instead of button with JavaScript redirect
                st.markdown(f"""
                    <div style="text-align: center; margin: 20px 0;">
                        <a href="{auth_url}" target="_self" style="
                            display: inline-block;
                            padding: 12px 24px;
                            background-color: #1f77b4;
                            color: white;
                            text-decoration: none;
                            border-radius: 5px;
                            font-weight: 600;
                            font-size: 16px;
                            transition: background-color 0.3s;
                        ">
                            🔑 Sign in with Google
                        </a>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.caption("🔒 Your credentials are never stored. Authentication is handled securely by Google.")

            # Debug information (can be removed later)
            with st.expander("🔧 Debug Information"):
                st.code(f"Redirect URI: {self.redirect_uri}")
                st.code(f"Client ID configured: {'Yes' if 'web' in st.secrets or os.path.exists(self.client_secrets_file) else 'No'}")

        except Exception as e:
            st.error(f"❌ Error generating login URL: {str(e)}")
            st.code(f"Redirect URI: {self.redirect_uri}")
            st.code(f"Error details: {type(e).__name__}: {str(e)}")

    def logout(self):
        """Log out the current user."""
        st.session_state.authenticated = False
        st.session_state.user_info = None
        st.session_state.auth_state = None

        # Clear chat history on logout for security
        if "messages" in st.session_state:
            st.session_state.messages = []
        if "chat_history" in st.session_state:
            st.session_state.chat_history = []

        st.rerun()

    def get_user_info(self) -> Optional[Dict[str, Any]]:
        """
        Get current authenticated user info.

        Returns:
            User info dictionary or None if not authenticated
        """
        if st.session_state.get("authenticated"):
            return st.session_state.get("user_info")
        return None

    def require_auth(self):
        """
        Decorator/function to require authentication.
        Call this at the start of your Streamlit app.
        """
        if not st.session_state.get("authenticated"):
            self.login()
            return False
        return True

    def show_user_widget(self):
        """Display user info widget in sidebar."""
        user_info = self.get_user_info()

        if not user_info:
            return

        with st.sidebar:
            st.markdown("---")
            st.markdown("### 👤 Logged in as:")

            col1, col2 = st.columns([1, 3])

            with col1:
                if user_info.get('picture'):
                    st.image(user_info['picture'], width=50)

            with col2:
                st.markdown(f"**{user_info['name']}**")
                st.caption(user_info['email'])

            if st.button("🚪 Logout", use_container_width=True):
                self.logout()


def get_auth_handler() -> GoogleOAuthHandler:
    """
    Get or create the global OAuth handler instance.

    Returns:
        GoogleOAuthHandler instance
    """
    if "oauth_handler" not in st.session_state:
        st.session_state.oauth_handler = GoogleOAuthHandler()

    return st.session_state.oauth_handler
