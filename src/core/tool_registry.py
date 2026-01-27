"""
Shared Tool Registry
Defines all PM Intelligence tools in one place.
Used by both MCP server and LangChain agent.
"""

from typing import Dict, List, Any


class ToolDefinition:
    """Standard tool definition used across interfaces."""
    
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler_func: str,  # Function name in handlers
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler_func = handler_func
    
    def to_mcp_schema(self) -> Dict[str, Any]:
        """Convert to MCP tool schema."""
        return {
            "type": "object",
            "required": self.parameters.get("required", []),
            "properties": self.parameters.get("properties", {})
        }
    
    def to_langchain_schema(self) -> Dict[str, Any]:
        """Convert to LangChain tool schema."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


# ==================== Tool Definitions ====================

PM_TOOLS = [
    ToolDefinition(
        name="initialize_pm_intelligence",
        description=(
            "Initialize PM Intelligence session. Provides guidelines for tool usage, "
            "source priorities, citation rules, and PM-specific analysis patterns. "
            "MUST be called once at the beginning of each session."
        ),
        parameters={
            "required": ["initialize_session"],
            "properties": {
                "initialize_session": {
                    "type": "boolean",
                    "description": "Flag to initialize the session. Must be set to true."
                }
            }
        },
        handler_func="get_pm_system_prompt"
    ),
    
    ToolDefinition(
        name="search_confluence",
        description=(
            "Search Confluence for internal documentation, project pages, and process guides. "
            "Best for: internal processes, best practices, detailed project documentation."
        ),
        parameters={
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for Confluence"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 10,
                    "minimum": 1
                },
                "space_filter": {
                    "type": "string",
                    "description": "Optional: Specific Confluence space to search"
                }
            }
        },
        handler_func="search_confluence"
    ),
    
    ToolDefinition(
        name="search_slack",
        description=(
            "Search Slack messages for team discussions, announcements, and real-time updates. "
            "Best for: latest updates, informal knowledge, team discussions, recent announcements."
        ),
        parameters={
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for Slack messages"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 10,
                    "minimum": 1
                },
                "channel_filter": {
                    "type": "string",
                    "description": "Optional: Specific channel to search"
                },
                "max_age_hours": {
                    "type": "integer",
                    "description": "Maximum age of messages in hours (default: 168 = 1 week)",
                    "default": 168
                }
            }
        },
        handler_func="search_slack"
    ),
    
    ToolDefinition(
        name="search_knowledge_base",
        description=(
            "Search knowledge base using vector similarity. Contains Incorta Community articles, "
            "official documentation, and support articles. "
            "Best for: product features, official documentation, authoritative product information."
        ),
        parameters={
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for knowledge base"
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5,
                    "minimum": 1
                }
            }
        },
        handler_func="search_knowledge_base"
    ),
    
    ToolDefinition(
        name="get_zendesk_schema",
        description=(
            "Get Zendesk schema details from Incorta (tables and columns). "
            "Call this before querying Zendesk data to understand available fields."
        ),
        parameters={
            "required": ["fetch_schema"],
            "properties": {
                "fetch_schema": {
                    "type": "boolean",
                    "description": "Flag to fetch schema details"
                }
            }
        },
        handler_func="get_zendesk_schema"
    ),
    
    ToolDefinition(
        name="query_zendesk",
        description=(
            "Execute SQL query on Zendesk data in Incorta. "
            "Best for: customer issues, support trends, pain point patterns. "
            "Must call get_zendesk_schema first to understand available fields."
        ),
        parameters={
            "required": ["spark_sql"],
            "properties": {
                "spark_sql": {
                    "type": "string",
                    "description": "Spark SQL query to execute"
                }
            }
        },
        handler_func="query_zendesk"
    ),
    
    ToolDefinition(
        name="get_jira_schema",
        description=(
            "Get Jira schema details from Incorta (tables and columns). "
            "Call this before querying Jira data to understand available fields."
        ),
        parameters={
            "required": ["fetch_schema"],
            "properties": {
                "fetch_schema": {
                    "type": "boolean",
                    "description": "Flag to fetch schema details"
                }
            }
        },
        handler_func="get_jira_schema"
    ),
    
    ToolDefinition(
        name="query_jira",
        description=(
            "Execute SQL query on Jira data in Incorta. "
            "Best for: development status, roadmap, feature progress, bug tracking. "
            "Must call get_jira_schema first to understand available fields."
        ),
        parameters={
            "required": ["spark_sql"],
            "properties": {
                "spark_sql": {
                    "type": "string",
                    "description": "Spark SQL query to execute"
                }
            }
        },
        handler_func="query_jira"
    ),
]


def get_tool_by_name(name: str) -> ToolDefinition:
    """Get tool definition by name."""
    for tool in PM_TOOLS:
        if tool.name == name:
            return tool
    raise ValueError(f"Tool not found: {name}")


def get_all_tool_names() -> List[str]:
    """Get list of all tool names."""
    return [tool.name for tool in PM_TOOLS]
