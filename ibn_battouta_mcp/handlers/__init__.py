"""Handlers package for MCP server."""

from .confluence_handler import search_confluence_pages
from .slack_handler import search_slack_simplified

__all__ = ["search_confluence_pages", "search_slack_simplified"]
