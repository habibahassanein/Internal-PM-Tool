"""Authentication module for Internal PM Tool."""

from .slack_auth import SlackOAuthHandler, get_slack_auth_handler
from .session_manager import SessionManager, get_session_manager

# Legacy Google auth support (deprecated)
try:
    from .google_auth import GoogleOAuthHandler, get_auth_handler
    _GOOGLE_AUTH_AVAILABLE = True
except ImportError:
    _GOOGLE_AUTH_AVAILABLE = False

__all__ = [
    'SlackOAuthHandler',
    'get_slack_auth_handler',
    'SessionManager',
    'get_session_manager'
]

# Add Google auth to exports if available
if _GOOGLE_AUTH_AVAILABLE:
    __all__.extend(['GoogleOAuthHandler', 'get_auth_handler'])
