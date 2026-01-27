"""
Tool Dispatcher
Routes tool calls to appropriate handlers.
Extracted from server.py for better organization.
"""

import json
import logging
from typing import Any, Dict

from tools.confluence_tool import search_confluence
from tools.slack_tool import search_slack
from tools.qdrant_tool import search_knowledge_base
from tools.incorta_tools import query_zendesk, query_jira, get_zendesk_schema, get_jira_schema
from tools.system_prompt_tool import get_pm_system_prompt

logger = logging.getLogger(__name__)


async def dispatch_tool_call(name: str, arguments: Dict[str, Any]) -> dict:
    """
    Dispatch tool call to appropriate handler.
    Returns raw data for Claude to process.
    
    Args:
        name: Tool name
        arguments: Tool arguments
    
    Returns:
        dict: Tool execution result
    
    Raises:
        ValueError: If tool not found
        Exception: If tool execution fails
    """
    logger.info(f"Dispatching tool: {name}")
    logger.debug(f"Arguments: {json.dumps(arguments, indent=2)}")
    
    # Map tool names to handlers
    handlers = {
        "initialize_pm_intelligence": get_pm_system_prompt,
        "search_confluence": search_confluence,
        "search_slack": search_slack,
        "search_knowledge_base": search_knowledge_base,
        "get_zendesk_schema": get_zendesk_schema,
        "query_zendesk": query_zendesk,
        "get_jira_schema": get_jira_schema,
        "query_jira": query_jira,
    }
    
    handler = handlers.get(name)
    if not handler:
        raise ValueError(f"Unknown tool: {name}")
    
    try:
        result = handler(arguments)
        logger.debug(f"Tool {name} completed successfully")
        return result
    except Exception as e:
        logger.exception(f"Error executing tool {name}: {e}")
        raise
