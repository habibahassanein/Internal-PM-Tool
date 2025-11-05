"""
Simple cookie-based authentication persistence for Streamlit.
"""

import streamlit as st
import hashlib
import hmac
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any


class CookieManager:
    """Manages authentication cookies for persistent login."""

    COOKIE_NAME = "pm_tool_auth"
    COOKIE_EXPIRY_DAYS = 7

    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize cookie manager.

        Args:
            secret_key: Secret key for signing cookies (use a secure random value)
        """
        self.secret_key = secret_key or "your-secret-key-change-in-production"

    def _sign_data(self, data: str) -> str:
        """Sign data with HMAC."""
        return hmac.new(
            self.secret_key.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()

    def _verify_signature(self, data: str, signature: str) -> bool:
        """Verify HMAC signature."""
        expected_signature = self._sign_data(data)
        return hmac.compare_digest(expected_signature, signature)

    def set_auth_cookie(self, user_info: Dict[str, Any]):
        """
        Set authentication cookie.

        Args:
            user_info: User information to store
        """
        # Create cookie data
        cookie_data = {
            'email': user_info.get('email'),
            'name': user_info.get('name'),
            'sub': user_info.get('sub'),
            'expires': (datetime.utcnow() + timedelta(days=self.COOKIE_EXPIRY_DAYS)).isoformat()
        }

        # Serialize and sign
        data_json = json.dumps(cookie_data)
        signature = self._sign_data(data_json)

        # Create cookie value
        cookie_value = f"{data_json}|{signature}"

        # Set cookie using JavaScript
        js_code = f"""
        <script>
            // Set cookie with expiry
            var expires = new Date();
            expires.setTime(expires.getTime() + ({self.COOKIE_EXPIRY_DAYS} * 24 * 60 * 60 * 1000));
            document.cookie = "{self.COOKIE_NAME}=" + encodeURIComponent("{cookie_value}") +
                            ";expires=" + expires.toUTCString() +
                            ";path=/;SameSite=Strict";
        </script>
        """
        st.markdown(js_code, unsafe_allow_html=True)

    def get_auth_cookie(self) -> Optional[Dict[str, Any]]:
        """
        Get and validate authentication cookie.

        Returns:
            User info if valid cookie exists, None otherwise
        """
        try:
            # Try to get cookie using query parameters (Streamlit workaround)
            # This is a simplified approach - in production, use a proper cookie library

            # For now, use session state as fallback
            if 'cookie_user_info' in st.session_state:
                return st.session_state.cookie_user_info

            return None

        except Exception:
            return None

    def clear_auth_cookie(self):
        """Clear authentication cookie."""
        js_code = f"""
        <script>
            document.cookie = "{self.COOKIE_NAME}=;expires=Thu, 01 Jan 1970 00:00:00 GMT;path=/";
        </script>
        """
        st.markdown(js_code, unsafe_allow_html=True)

        if 'cookie_user_info' in st.session_state:
            del st.session_state.cookie_user_info


def get_cookie_manager() -> CookieManager:
    """
    Get or create the global cookie manager instance.

    Returns:
        CookieManager instance
    """
    if "cookie_manager" not in st.session_state:
        st.session_state.cookie_manager = CookieManager()

    return st.session_state.cookie_manager
