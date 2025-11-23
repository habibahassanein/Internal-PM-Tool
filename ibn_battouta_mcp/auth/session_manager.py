"""
Session Manager for Multi-User MCP Server

Manages user sessions and Slack tokens for multiple concurrent users.
Each user authenticates once via OAuth and receives a session ID.
"""

import secrets
import time
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class UserSession:
    """Represents a user session with their Slack credentials."""
    session_id: str
    slack_token: str
    user_name: str
    user_email: str
    created_at: float
    last_accessed: float


class SessionManager:
    """
    Thread-safe session manager for storing user Slack tokens.

    Sessions expire after 30 days of inactivity.
    """

    def __init__(self, session_ttl: int = 30 * 24 * 60 * 60):  # 30 days default
        self._sessions: Dict[str, UserSession] = {}
        self._session_ttl = session_ttl

    def create_session(
        self,
        slack_token: str,
        user_name: str = "Unknown",
        user_email: str = ""
    ) -> str:
        """
        Create a new session and return the session ID.

        Args:
            slack_token: User's Slack OAuth token
            user_name: User's display name
            user_email: User's email

        Returns:
            session_id: Unique session identifier
        """
        session_id = secrets.token_urlsafe(32)
        now = time.time()

        session = UserSession(
            session_id=session_id,
            slack_token=slack_token,
            user_name=user_name,
            user_email=user_email,
            created_at=now,
            last_accessed=now
        )

        self._sessions[session_id] = session
        logger.info(f"Created session for user: {user_name} ({user_email})")

        return session_id

    def get_slack_token(self, session_id: str) -> Optional[str]:
        """
        Retrieve Slack token for a session.

        Args:
            session_id: Session identifier

        Returns:
            Slack token if session is valid, None otherwise
        """
        if not session_id:
            return None

        session = self._sessions.get(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id[:8]}...")
            return None

        # Check if session expired
        now = time.time()
        if now - session.last_accessed > self._session_ttl:
            logger.warning(f"Session expired for user: {session.user_name}")
            del self._sessions[session_id]
            return None

        # Update last accessed time
        session.last_accessed = now

        return session.slack_token

    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get full session information."""
        if not session_id:
            return None

        session = self._sessions.get(session_id)
        if not session:
            return None

        # Check expiration
        now = time.time()
        if now - session.last_accessed > self._session_ttl:
            del self._sessions[session_id]
            return None

        session.last_accessed = now
        return session

    def revoke_session(self, session_id: str) -> bool:
        """
        Revoke/delete a session.

        Args:
            session_id: Session to revoke

        Returns:
            True if session was deleted, False if not found
        """
        if session_id in self._sessions:
            user_name = self._sessions[session_id].user_name
            del self._sessions[session_id]
            logger.info(f"Revoked session for user: {user_name}")
            return True
        return False

    def cleanup_expired_sessions(self) -> int:
        """
        Remove all expired sessions.

        Returns:
            Number of sessions removed
        """
        now = time.time()
        expired = [
            sid for sid, session in self._sessions.items()
            if now - session.last_accessed > self._session_ttl
        ]

        for sid in expired:
            del self._sessions[sid]

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")

        return len(expired)

    def get_active_sessions_count(self) -> int:
        """Get number of active sessions."""
        return len(self._sessions)

    def list_sessions(self) -> list[Dict]:
        """
        List all active sessions (for admin purposes).

        Returns:
            List of session info (without tokens)
        """
        now = time.time()
        return [
            {
                "session_id": session.session_id[:16] + "...",  # Truncated for security
                "user_name": session.user_name,
                "user_email": session.user_email,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(session.created_at)),
                "last_accessed": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(session.last_accessed)),
                "age_hours": round((now - session.created_at) / 3600, 1)
            }
            for session in self._sessions.values()
        ]


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
