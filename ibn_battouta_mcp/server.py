
import json
import logging
import contextlib
import os
from collections.abc import AsyncIterator
from typing import Any, Dict

from dotenv import load_dotenv
import click
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.responses import Response, JSONResponse, RedirectResponse, HTMLResponse
from starlette.routing import Mount, Route
from starlette.types import Receive, Scope, Send
from starlette.requests import Request as StarletteRequest
import requests
from slack_sdk.web import WebClient

from context.user_context import user_context
from tools.confluence_tool import search_confluence
from tools.slack_tool import search_slack
from tools.qdrant_tool import search_knowledge_base
from tools.incorta_tools import query_zendesk, query_jira, get_zendesk_schema, get_jira_schema
from tools.system_prompt_tool import get_pm_system_prompt
from auth.session_manager import get_session_manager

logger = logging.getLogger("ibn-battouta-mcp-server")
logging.basicConfig(level=logging.INFO)

IBN_BATTOUTA_MCP_PORT = 8080  # Different from Incorta MCP (8080)


@click.command()
@click.option("--port", default=IBN_BATTOUTA_MCP_PORT, help="Port to listen on for HTTP")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--json-response", is_flag=True, default=True, help="Enable JSON responses")
def main(port: int, log_level: str, json_response: bool) -> int:
    """Ibn Battouta MCP server with SSE + StreamableHTTP transports."""
    logging.getLogger().setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Ensure values from .env are loaded into process environment before we access them.
    # This allows local development to rely on .env while still letting real env vars override.
    load_dotenv()

    app = Server("Ibn Battouta - PM Intelligence")

    # ----------------------------- Schema Helpers -----------------------------#
    def make_boolean_flag_schema(flag_name: str, description: str) -> dict:
        """Helper to create boolean flag input schema."""
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
        """Helper to create SQL query input schema."""
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
        """Helper to create search input schema with optional parameters."""
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
        """
        Lists all available tools for the MCP Client (Claude).
        No Gemini needed - Claude will handle synthesis and citations.
        """
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
            # types.Tool(
            #     name="search_slack",
            #     description=(
            #         "Search Slack messages for team discussions, announcements, and real-time updates. "
            #         "Best for: latest updates, informal knowledge, team discussions, recent announcements. "
            #         "Returns: Message text, username, channel, timestamp, permalink with source='slack'."
            #     ),
            #     inputSchema=make_search_schema(
            #         "Search query for Slack messages",
            #         {
            #             "max_results": {
            #                 "type": "integer",
            #                 "description": "Maximum number of results to return",
            #                 "default": 10,
            #                 "minimum": 1
            #             },
            #             "channel_filter": {
            #                 "type": "string",
            #                 "description": "Optional: Specific channel to search"
            #             },
            #             "max_age_hours": {
            #                 "type": "integer",
            #                 "description": "Maximum age of messages in hours (default: 168 = 1 week)",
            #                 "default": 168
            #             }
            #         }
            #     ),
            # ),
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
        """
        Calls a tool by its name with the provided arguments.
        Returns raw data - Claude will synthesize and create citations.
        """
        logger.info(f"call_tool: {name}")
        logger.debug(f"raw arguments: {json.dumps(arguments, indent=2)}")

        try:
            if name == "initialize_pm_intelligence":
                result = get_pm_system_prompt(arguments)
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

            elif name == "search_confluence":
                result = search_confluence(arguments)
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

            # elif name == "search_slack":
            #     result = search_slack(arguments)
            #     return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

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


    # ------------------------------- Transports ------------------------------ #
    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        """SSE transport endpoint."""
        logger.info("Handling SSE connection")

        try:
            async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
                await app.run(streams[0], streams[1], app.create_initialization_options())
        except Exception as e:
            logger.exception(f"Error handling SSE connection: {e}")

        return Response()

    async def handle_root(request):
        """Simple health/status endpoint for browsers."""
        return JSONResponse(
            {
                "service": "Ibn Battouta MCP Server",
                "status": "ok",
                "transports": {
                    "sse": "/sse",
                    "streamable_http": "/mcp",
                },
                "tools": [
                    "initialize_pm_intelligence",
                    "search_confluence",
                    "search_slack",
                    "search_knowledge_base",
                    "get_zendesk_schema",
                    "query_zendesk",
                    "get_jira_schema",
                    "query_jira",
                ],
            }
        )

    session_manager = StreamableHTTPSessionManager(
        app=app,
        event_store=None,
        json_response=json_response,
        stateless=True,
    )

    async def handle_streamable_http(scope: Scope, receive: Receive, send: Send) -> None:
        """
        Streamable HTTP transport endpoint.
        Accepts credentials via headers for per-request auth.
        Supports session-based authentication for Slack.
        """
        logger.info("Handling StreamableHTTP request")
        headers = {k.decode("utf-8"): v.decode("utf-8") for k, v in scope.get("headers", [])}

        # Get session ID from headers
        session_id = headers.get("session-id")

        # Retrieve Slack token from session if session ID provided
        slack_token = None
        if session_id:
            session_mgr = get_session_manager()
            slack_token = session_mgr.get_slack_token(session_id)
            if slack_token:
                logger.info(f"Retrieved Slack token from session: {session_id[:16]}...")
            else:
                logger.warning(f"Invalid or expired session: {session_id[:16]}...")

        # Fallback to direct token in header (for backward compatibility)
        if not slack_token:
            slack_token = headers.get("slack-token")

        # Load unified credentials from environment variables
        incorta_env_url = headers.get("incorta-env-url") or os.getenv("INCORTA_ENV_URL")
        incorta_tenant = headers.get("incorta-tenant") or os.getenv("INCORTA_TENANT")
        incorta_access_token = headers.get("incorta-access-token") or os.getenv("INCORTA_MCP_TOKEN") or os.getenv("PAT")
        incorta_username = headers.get("incorta-username") or os.getenv("INCORTA_USERNAME")
        incorta_password = headers.get("incorta-password") or os.getenv("INCORTA_PASSWORD")
        incorta_sqlx_host = headers.get("incorta-sqlx-host") or os.getenv("INCORTA_SQLX_HOST")

        qdrant_url = headers.get("qdrant-url") or os.getenv("QDRANT_URL")
        qdrant_api_key = headers.get("qdrant-api-key") or os.getenv("QDRANT_API_KEY")

        confluence_url = headers.get("confluence-url") or os.getenv("CONFLUENCE_URL")
        confluence_token = headers.get("confluence-token") or os.getenv("CONFLUENCE_API_TOKEN")

        # Set user context from headers and environment
        user_context.set({
            # Incorta credentials (unified for all users)
            "incorta_env_url": incorta_env_url,
            "incorta_tenant": incorta_tenant,
            "incorta_access_token": incorta_access_token,
            "incorta_username": incorta_username,
            "incorta_password": incorta_password,
            "incorta_sqlx_host": incorta_sqlx_host,

            # Qdrant credentials (unified for all users)
            "qdrant_url": qdrant_url,
            "qdrant_api_key": qdrant_api_key,

            # Confluence credentials (unified for all users)
            "confluence_url": confluence_url,
            "confluence_token": confluence_token,

            # Slack credentials (per-user from session)
            "slack_token": slack_token,

            # Session info
            "session_id": session_id,
        })

        logger.info(f"Context set with {len([k for k, v in user_context.get().items() if v])} credentials")

        try:
            await session_manager.handle_request(scope, receive, send)
        except Exception as e:
            logger.exception(f"Error handling StreamableHTTP request: {e}")

    # ------------------------------- OAuth Endpoints ----------------------------- #
    async def handle_auth_start(request: StarletteRequest):
        """
        Start Slack OAuth flow.
        Redirects user to Slack authorization page.
        """
        # Get OAuth config from environment
        client_id = os.getenv("SLACK_CLIENT_ID")
        redirect_uri = os.getenv("SLACK_REDIRECT_URI", f"http://localhost:{port}/auth/callback")

        if not client_id:
            return HTMLResponse(
                "<h1>Error</h1><p>SLACK_CLIENT_ID not configured in environment</p>",
                status_code=500
            )

        # Scopes needed for user token
        scopes = [
            "channels:history",
            "channels:read",
            "groups:history",
            "groups:read",
            "search:read",
            "users:read",
        ]

        auth_url = (
            f"https://slack.com/oauth/v2/authorize?"
            f"client_id={client_id}&"
            f"user_scope={','.join(scopes)}&"
            f"redirect_uri={redirect_uri}"
        )

        return RedirectResponse(auth_url)

    async def handle_auth_callback(request: StarletteRequest):
        """
        OAuth callback endpoint.
        Exchanges code for token and creates session.
        """
        # Get code from query params
        code = request.query_params.get("code")
        error = request.query_params.get("error")

        if error:
            return HTMLResponse(
                f"<h1>Authentication Failed</h1><p>Error: {error}</p>",
                status_code=400
            )

        if not code:
            return HTMLResponse(
                "<h1>Authentication Failed</h1><p>No authorization code received</p>",
                status_code=400
            )

        # Exchange code for token
        client_id = os.getenv("SLACK_CLIENT_ID")
        client_secret = os.getenv("SLACK_CLIENT_SECRET")
        redirect_uri = os.getenv("SLACK_REDIRECT_URI", f"http://localhost:{port}/auth/callback")

        if not client_id or not client_secret:
            return HTMLResponse(
                "<h1>Server Error</h1><p>OAuth credentials not configured</p>",
                status_code=500
            )

        try:
            response = requests.post(
                "https://slack.com/api/oauth.v2.access",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": code,
                    "redirect_uri": redirect_uri,
                },
                timeout=15,
            )
            data = response.json()

            if not data.get("ok"):
                error_msg = data.get("error", "Unknown error")
                return HTMLResponse(
                    f"<h1>OAuth Failed</h1><p>Error: {error_msg}</p>",
                    status_code=400
                )

            # Extract user token
            slack_token = data["authed_user"]["access_token"]

            # Get user info
            client = WebClient(token=slack_token)
            user_info = client.users_identity()
            user_name = user_info["user"]["name"]
            user_email = user_info["user"].get("email", "")

            # Create session
            session_mgr = get_session_manager()
            session_id = session_mgr.create_session(
                slack_token=slack_token,
                user_name=user_name,
                user_email=user_email
            )

            # Return success page with session ID
            return HTMLResponse(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Authentication Successful</title>
                    <style>
                        body {{
                            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                            max-width: 600px;
                            margin: 50px auto;
                            padding: 20px;
                            background: #f5f5f5;
                        }}
                        .container {{
                            background: white;
                            padding: 30px;
                            border-radius: 8px;
                            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                        }}
                        h1 {{ color: #2ecc71; }}
                        .session-id {{
                            background: #f8f9fa;
                            padding: 15px;
                            border-radius: 4px;
                            border-left: 4px solid #2ecc71;
                            font-family: monospace;
                            word-break: break-all;
                            margin: 20px 0;
                        }}
                        .instructions {{
                            background: #e3f2fd;
                            padding: 15px;
                            border-radius: 4px;
                            margin: 20px 0;
                        }}
                        code {{
                            background: #f5f5f5;
                            padding: 2px 6px;
                            border-radius: 3px;
                            font-family: monospace;
                        }}
                        .copy-btn {{
                            background: #4A154B;
                            color: white;
                            border: none;
                            padding: 10px 20px;
                            border-radius: 4px;
                            cursor: pointer;
                            font-size: 14px;
                        }}
                        .copy-btn:hover {{ background: #611f69; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>✅ Authentication Successful!</h1>
                        <p>Welcome, <strong>{user_name}</strong>!</p>

                        <h2>Your Session ID:</h2>
                        <div class="session-id" id="sessionId">{session_id}</div>
                        <button class="copy-btn" onclick="copySessionId()">Copy Session ID</button>

                        <div class="instructions">
                            <h3>Next Steps:</h3>
                            <ol>
                                <li>Copy your session ID above</li>
                                <li>Add it to your Claude Desktop config at:<br>
                                    <code>~/Library/Application Support/Claude/claude_desktop_config.json</code>
                                </li>
                                <li>Use this configuration:</li>
                            </ol>
                            <pre style="background: #2d2d2d; color: #f8f8f2; padding: 15px; border-radius: 4px; overflow-x: auto;">{{
  "mcpServers": {{
    "ibn-battouta": {{
      "url": "http://localhost:{port}/mcp",
      "transport": "http",
      "headers": {{
        "session-id": "{session_id}"
      }}
    }}
  }}
}}</pre>
                            <li>Restart Claude Desktop</li>
                        </div>

                        <p><small>Session expires after 30 days of inactivity.</small></p>
                    </div>

                    <script>
                        function copySessionId() {{
                            const sessionId = document.getElementById('sessionId').textContent;
                            navigator.clipboard.writeText(sessionId).then(() => {{
                                const btn = document.querySelector('.copy-btn');
                                btn.textContent = '✓ Copied!';
                                setTimeout(() => {{ btn.textContent = 'Copy Session ID'; }}, 2000);
                            }});
                        }}
                    </script>
                </body>
                </html>
            """)

        except Exception as e:
            logger.exception(f"Error during OAuth callback: {e}")
            return HTMLResponse(
                f"<h1>Error</h1><p>Failed to complete authentication: {str(e)}</p>",
                status_code=500
            )

    async def handle_auth_status(request: StarletteRequest):
        """Check session status (for debugging)."""
        session_id = request.query_params.get("session_id", "")

        if not session_id:
            return JSONResponse({"error": "session_id parameter required"}, status_code=400)

        session_mgr = get_session_manager()
        session = session_mgr.get_session(session_id)

        if not session:
            return JSONResponse({"valid": False, "message": "Session not found or expired"})

        return JSONResponse({
            "valid": True,
            "user_name": session.user_name,
            "user_email": session.user_email,
            "created_at": session.created_at,
            "last_accessed": session.last_accessed
        })

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        """Application lifespan management."""
        async with session_manager.run():
            logger.info("Ibn Battouta MCP Server started with dual transports!")
            logger.info(f"OAuth authentication URL: http://localhost:{port}/auth/slack")
            try:
                yield
            finally:
                logger.info("Ibn Battouta MCP Server shutting down...")

    starlette_app = Starlette(
        debug=True,
        routes=[
            Route("/", endpoint=handle_root, methods=["GET"]),
            Route("/sse", endpoint=handle_sse, methods=["GET"]),
            Route("/auth/slack", endpoint=handle_auth_start, methods=["GET"]),
            Route("/auth/callback", endpoint=handle_auth_callback, methods=["GET"]),
            Route("/auth/status", endpoint=handle_auth_status, methods=["GET"]),
            Mount("/messages/", app=sse.handle_post_message),
            Mount("/mcp", app=handle_streamable_http),
        ],
        lifespan=lifespan,
    )

    logger.info(f"Ibn Battouta MCP Server starting on port {port} with dual transports:")
    logger.info(f"  - SSE endpoint: http://localhost:{port}/sse")
    logger.info(f"  - StreamableHTTP endpoint: http://localhost:{port}/mcp")
    logger.info(f"")
    logger.info(f"Available tools:")
    logger.info(f"  1. initialize_pm_intelligence - System prompt and guidelines")
    logger.info(f"  2. search_confluence - Search internal documentation")
    logger.info(f"  3. search_slack - Search team discussions")
    logger.info(f"  4. search_knowledge_base - Vector search (Incorta docs)")
    logger.info(f"  5. get_zendesk_schema - Get Zendesk schema structure")
    logger.info(f"  6. query_zendesk - Query customer support data")
    logger.info(f"  7. get_jira_schema - Get Jira schema structure")
    logger.info(f"  8. query_jira - Query development data")

    import uvicorn
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)
    return 0


if __name__ == "__main__":
    main()
