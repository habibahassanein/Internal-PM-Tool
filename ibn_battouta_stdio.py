#!/usr/bin/env python
"""
Wrapper script for Ibn Battouta MCP Server - StdIO Transport
Properly sets up Python path and environment for Claude Desktop.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Add source directory to path for proper imports
SOURCE_DIR = Path(__file__).parent
sys.path.insert(0, str(SOURCE_DIR))

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

# Now we can import with relative paths since SOURCE_DIR is in sys.path
from ibn_battouta_mcp.context.user_context import user_context
from ibn_battouta_mcp.tools.confluence_tool import search_confluence
from ibn_battouta_mcp.tools.qdrant_tool import search_knowledge_base
from ibn_battouta_mcp.tools.incorta_tools import query_zendesk, query_jira, get_zendesk_schema, get_jira_schema
from ibn_battouta_mcp.tools.system_prompt_tool import get_pm_system_prompt

# Configure logging to stderr (stdout is used for MCP protocol)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("ibn-battouta-stdio")


async def main():
    """Run the MCP server with StdIO transport."""
    logger.info("Starting Ibn Battouta MCP Server (StdIO mode)")

    # Load credentials from environment variables
    user_context.set({
        # Incorta credentials
        "incorta_env_url": os.getenv("INCORTA_ENV_URL"),
        "incorta_tenant": os.getenv("INCORTA_TENANT"),
        "incorta_username": os.getenv("INCORTA_USERNAME"),
        "incorta_password": os.getenv("INCORTA_PASSWORD"),
        "incorta_sqlx_host": os.getenv("INCORTA_SQLX_HOST"),

        # Qdrant credentials
        "qdrant_url": os.getenv("QDRANT_URL"),
        "qdrant_api_key": os.getenv("QDRANT_API_KEY"),

        # Confluence credentials
        "confluence_url": os.getenv("CONFLUENCE_URL"),
        "confluence_token": os.getenv("CONFLUENCE_API_TOKEN"),
    })

    logger.info(f"Loaded {len([k for k, v in user_context.get().items() if v])} credentials from environment")

    app = Server("Ibn Battouta - PM Intelligence")

    # ----------------------------- Tool Registry -----------------------------#
    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        """Lists all available tools for Claude Desktop."""
        return [
            types.Tool(
                name="initialize_pm_intelligence",
                description=(
                    "MUST be called once at the beginning of each session to establish proper context "
                    "for multi-source PM intelligence. Provides guidelines for tool usage, source priorities, "
                    "citation rules, and PM-specific analysis patterns."
                ),
                inputSchema={
                    "type": "object",
                    "required": ["initialize_session"],
                    "properties": {
                        "initialize_session": {
                            "type": "boolean",
                            "description": "Flag to initialize the session. Must be set to true."
                        }
                    }
                },
            ),
            types.Tool(
                name="search_confluence",
                description=(
                    "Search Confluence pages for internal documentation, project pages, and process guides. "
                    "Best for: internal processes, best practices, detailed project documentation. "
                    "Returns: Page titles, URLs, excerpts with source='confluence'."
                ),
                inputSchema={
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for Confluence"
                        },
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
                },
            ),
            types.Tool(
                name="search_knowledge_base",
                description=(
                    "Search the knowledge base using vector similarity. Contains Incorta Community articles, "
                    "official documentation, and support articles. "
                    "Best for: product features, official documentation, authoritative product information. "
                    "Returns: Article titles, URLs, text excerpts, relevance scores with source='knowledge_base'."
                ),
                inputSchema={
                    "type": "object",
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
            ),
            types.Tool(
                name="get_zendesk_schema",
                description=(
                    "Get Zendesk schema details from Incorta (tables and columns). "
                    "Call this before querying Zendesk data to understand available fields. "
                    "Returns: Schema structure with table names and column definitions."
                ),
                inputSchema={
                    "type": "object",
                    "required": ["fetch_schema"],
                    "properties": {
                        "fetch_schema": {
                            "type": "boolean",
                            "description": "Flag to fetch schema details"
                        }
                    }
                },
            ),
            types.Tool(
                name="query_zendesk",
                description=(
                    "Execute SQL query on Zendesk data in Incorta. "
                    "Best for: customer issues, support trends, pain point patterns. "
                    "Must call get_zendesk_schema first to understand available fields. "
                    "Returns: Query results with columns and rows, source='zendesk'."
                ),
                inputSchema={
                    "type": "object",
                    "required": ["spark_sql"],
                    "properties": {
                        "spark_sql": {
                            "type": "string",
                            "description": "Spark SQL query to execute on Zendesk schema"
                        }
                    }
                },
            ),
            types.Tool(
                name="get_jira_schema",
                description=(
                    "Get Jira schema details from Incorta (tables and columns). "
                    "Call this before querying Jira data to understand available fields. "
                    "Returns: Schema structure with table names and column definitions."
                ),
                inputSchema={
                    "type": "object",
                    "required": ["fetch_schema"],
                    "properties": {
                        "fetch_schema": {
                            "type": "boolean",
                            "description": "Flag to fetch schema details"
                        }
                    }
                },
            ),
            types.Tool(
                name="query_jira",
                description=(
                    "Execute SQL query on Jira data in Incorta. "
                    "Best for: development status, roadmap, feature progress, bug tracking. "
                    "Must call get_jira_schema first to understand available fields. "
                    "Returns: Query results with columns and rows, source='jira'."
                ),
                inputSchema={
                    "type": "object",
                    "required": ["spark_sql"],
                    "properties": {
                        "spark_sql": {
                            "type": "string",
                            "description": "Spark SQL query to execute on Jira schema"
                        }
                    }
                },
            ),
        ]

    # ---------------------------- Tool Dispatcher ----------------------------#
    @app.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> list[types.TextContent]:
        """Calls a tool by name with the provided arguments."""
        logger.info(f"Calling tool: {name}")
        logger.debug(f"Arguments: {json.dumps(arguments, indent=2)}")

        try:
            if name == "initialize_pm_intelligence":
                result = get_pm_system_prompt(arguments)
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

            elif name == "search_confluence":
                result = search_confluence(arguments)
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

            elif name == "search_knowledge_base":
                result = search_knowledge_base(arguments)
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

            elif name == "get_zendesk_schema":
                result = get_zendesk_schema(arguments)
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

            elif name == "query_zendesk":
                result = query_zendesk(arguments)
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

            elif name == "get_jira_schema":
                result = get_jira_schema(arguments)
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

            elif name == "query_jira":
                result = query_jira(arguments)
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

            else:
                return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

        except Exception as e:
            logger.exception(f"Error executing tool {name}: {e}")
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    # Run the StdIO server
    logger.info("Server ready, listening on stdin/stdout")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
