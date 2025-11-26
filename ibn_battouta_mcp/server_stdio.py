
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

# Add script directory to Python path for relative imports
# This ensures imports work regardless of current working directory
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Load .env from the script's directory (not cwd)
from dotenv import load_dotenv
load_dotenv(SCRIPT_DIR / ".env")

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from context.user_context import user_context
from tools.confluence_tool import search_confluence
from tools.slack_tool import search_slack
from tools.qdrant_tool import search_knowledge_base
from tools.incorta_tools import query_zendesk, query_jira, get_zendesk_schema, get_jira_schema
from tools.system_prompt_tool import get_pm_system_prompt

# Configure logging to stderr (stdout is used for MCP protocol)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("ibn-battouta-mcp-stdio")


def create_server() -> Server:
    """Create and configure the MCP server."""

    # Set user context from environment variables (already loaded at module level)
    user_context.set({
        # Incorta credentials
        "incorta_env_url": os.getenv("INCORTA_ENV_URL"),
        "incorta_tenant": os.getenv("INCORTA_TENANT"),
        "incorta_access_token": os.getenv("INCORTA_MCP_TOKEN") or os.getenv("PAT"),
        "incorta_username": os.getenv("INCORTA_USERNAME"),
        "incorta_password": os.getenv("INCORTA_PASSWORD"),
        "incorta_sqlx_host": os.getenv("INCORTA_SQLX_HOST"),

        # Qdrant credentials
        "qdrant_url": os.getenv("QDRANT_URL"),
        "qdrant_api_key": os.getenv("QDRANT_API_KEY"),

        # Confluence credentials
        "confluence_url": os.getenv("CONFLUENCE_URL"),
        "confluence_token": os.getenv("CONFLUENCE_API_TOKEN"),

        # Slack credentials
        "slack_token": os.getenv("SLACK_TOKEN"),
    })

    app = Server("Ibn Battouta - PM Intelligence")

    # ----------------------------- Schema Helpers -----------------------------#
    def make_boolean_flag_schema(flag_name: str, description: str) -> dict:
        return {
            "type": "object",
            "required": [flag_name],
            "properties": {
                flag_name: {
                    "type": "boolean",
                    "description": description
                }
            }
        }

    def make_sql_query_schema() -> dict:
        return {
            "type": "object",
            "required": ["spark_sql"],
            "properties": {
                "spark_sql": {
                    "type": "string",
                    "description": "Spark SQL query to execute"
                }
            }
        }

    def make_search_schema(query_desc: str, optional_params: dict = None) -> dict:
        properties = {
            "query": {
                "type": "string",
                "description": query_desc
            }
        }
        if optional_params:
            properties.update(optional_params)

        return {
            "type": "object",
            "required": ["query"],
            "properties": properties
        }

    # ----------------------------- Tool Registry -----------------------------#
    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="initialize_pm_intelligence",
                description=(
                    "MUST be called once at the beginning of each session to establish proper context "
                    "for multi-source PM intelligence. Provides guidelines for tool usage, source priorities, "
                    "citation rules, and PM-specific analysis patterns."
                ),
                inputSchema=make_boolean_flag_schema(
                    "initialize_session",
                    "Flag to initialize the session. Must be set to true."
                ),
            ),
            types.Tool(
                name="search_confluence",
                description=(
                    "Search Confluence pages for internal documentation, project pages, and process guides. "
                    "Best for: internal processes, best practices, detailed project documentation. "
                    "Returns: Page titles, URLs, excerpts with source='confluence'."
                ),
                inputSchema=make_search_schema(
                    "Search query for Confluence",
                    {
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 10,
                            "minimum": 1
                        },
                        "space_filter": {
                            "type": "string",
                            "description": "Optional: Specific Confluence space to search"
                        }
                    }
                ),
            ),
            types.Tool(
                name="search_slack",
                description=(
                    "Search Slack messages for team discussions, announcements, and real-time updates. "
                    "Best for: latest updates, informal knowledge, team discussions, recent announcements. "
                    "Returns: Message text, username, channel, timestamp, permalink with source='slack'."
                ),
                inputSchema=make_search_schema(
                    "Search query for Slack messages",
                    {
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
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
                ),
            ),
            types.Tool(
                name="search_knowledge_base",
                description=(
                    "Search the knowledge base using vector similarity. Contains Incorta Community articles, "
                    "official documentation, and support articles. "
                    "Best for: product features, official documentation, authoritative product information. "
                    "Returns: Article titles, URLs, text excerpts, relevance scores with source='knowledge_base'."
                ),
                inputSchema=make_search_schema(
                    "Search query for knowledge base",
                    {
                        "limit": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "default": 5,
                            "minimum": 1
                        }
                    }
                ),
            ),
            types.Tool(
                name="get_zendesk_schema",
                description=(
                    "Get Zendesk schema details from Incorta (tables and columns). "
                    "Call this before querying Zendesk data to understand available fields. "
                    "Returns: Schema structure with table names and column definitions."
                ),
                inputSchema=make_boolean_flag_schema("fetch_schema", "Flag to fetch schema details"),
            ),
            types.Tool(
                name="query_zendesk",
                description=(
                    "Execute SQL query on Zendesk data in Incorta. "
                    "Best for: customer issues, support trends, pain point patterns. "
                    "Must call get_zendesk_schema first to understand available fields. "
                    "Returns: Query results with columns and rows, source='zendesk'."
                ),
                inputSchema=make_sql_query_schema(),
            ),
            types.Tool(
                name="get_jira_schema",
                description=(
                    "Get Jira schema details from Incorta (tables and columns). "
                    "Call this before querying Jira data to understand available fields. "
                    "Returns: Schema structure with table names and column definitions."
                ),
                inputSchema=make_boolean_flag_schema("fetch_schema", "Flag to fetch schema details"),
            ),
            types.Tool(
                name="query_jira",
                description=(
                    "Execute SQL query on Jira data in Incorta. "
                    "Best for: development status, roadmap, feature progress, bug tracking. "
                    "Must call get_jira_schema first to understand available fields. "
                    "Returns: Query results with columns and rows, source='jira'."
                ),
                inputSchema=make_sql_query_schema(),
            ),
        ]

    # ---------------------------- Tool Dispatcher ----------------------------#
    @app.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> list[types.TextContent]:
        logger.info(f"call_tool: {name}")
        logger.debug(f"raw arguments: {json.dumps(arguments, indent=2)}")

        try:
            if name == "initialize_pm_intelligence":
                result = get_pm_system_prompt(arguments)
            elif name == "search_confluence":
                result = search_confluence(arguments)
            elif name == "search_slack":
                result = search_slack(arguments)
            elif name == "search_knowledge_base":
                result = search_knowledge_base(arguments)
            elif name == "get_zendesk_schema":
                result = get_zendesk_schema(arguments)
            elif name == "query_zendesk":
                result = query_zendesk(arguments)
            elif name == "get_jira_schema":
                result = get_jira_schema(arguments)
            elif name == "query_jira":
                result = query_jira(arguments)
            else:
                return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            logger.exception(f"Error executing tool {name}: {e}")
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    return app


async def main():
    """Run the server using stdio transport."""
    logger.info("Starting Ibn Battouta MCP Server (stdio transport)")

    app = create_server()

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
