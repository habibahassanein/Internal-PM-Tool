
import json
import logging
import contextlib
import os
import sys
from collections.abc import AsyncIterator
from typing import Any, Dict

# Add parent directory to path for src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from auth.session_manager import get_session_manager
from tool_dispatcher import dispatch_tool_call
from src.core.tool_registry import PM_TOOLS, get_all_tool_names

logger = logging.getLogger("ibn-battouta-mcp-server")
logging.basicConfig(level=logging.INFO)

IBN_BATTOUTA_MCP_PORT = 8080


@click.command()
@click.option("--port", default=IBN_BATTOUTA_MCP_PORT, help="Port to listen on for HTTP")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--json-response", is_flag=True, default=True, help="Enable JSON responses")
def main(port: int, log_level: str, json_response: bool) -> int:
    """Ibn Battouta MCP server with SSE + StreamableHTTP transports."""
    logging.getLogger().setLevel(getattr(logging, log_level.upper(), logging.INFO))
    load_dotenv()

    app = Server("Ibn Battouta - PM Intelligence")

    # ----------------------------- Tool Registry -----------------------------#
    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        """List all available tools from shared registry."""
        return [
            types.Tool(
                name=tool.name,
                description=tool.description,
                inputSchema=tool.to_mcp_schema()
            )
            for tool in PM_TOOLS
        ]

    # ---------------------------- Tool Dispatcher ----------------------------#
    @app.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> list[types.TextContent]:
        """Dispatch tool calls using shared dispatcher."""
        try:
            result = await dispatch_tool_call(name, arguments)
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
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
        """Simple health/status endpoint."""
        return JSONResponse({
            "service": "Ibn Battouta MCP Server",
            "status": "ok",
            "transports": {
                "sse": "/sse",
                "streamable_http": "/mcp",
            },
            "tools": get_all_tool_names(),
        })

    session_manager = StreamableHTTPSessionManager(
        app=app,
        event_store=None,
        json_response=json_response,
        stateless=True,
    )

    async def handle_streamable_http(scope: Scope, receive: Receive, send: Send) -> None:
        """StreamableHTTP transport with session-based Slack auth."""
        logger.info("Handling StreamableHTTP request")
        headers = {k.decode("utf-8"): v.decode("utf-8") for k, v in scope.get("headers", [])}

        # Get session ID and retrieve Slack token
        session_id = headers.get("session-id")
        slack_token = None
        if session_id:
            session_mgr = get_session_manager()
            slack_token = session_mgr.get_slack_token(session_id)
            if slack_token:
                logger.info(f"Retrieved Slack token from session: {session_id[:16]}...")
            else:
                logger.warning(f"Invalid or expired session: {session_id[:16]}...")

        # Fallback to direct token in header
        if not slack_token:
            slack_token = headers.get("slack-token")

        # Load unified credentials from headers or environment
        user_context.set({
            # Incorta credentials
            "incorta_env_url": headers.get("incorta-env-url") or os.getenv("INCORTA_ENV_URL"),
            "incorta_tenant": headers.get("incorta-tenant") or os.getenv("INCORTA_TENANT"),
            "incorta_access_token": headers.get("incorta-access-token") or os.getenv("INCORTA_MCP_TOKEN") or os.getenv("PAT"),
            "incorta_username": headers.get("incorta-username") or os.getenv("INCORTA_USERNAME"),
            "incorta_password": headers.get("incorta-password") or os.getenv("INCORTA_PASSWORD"),
            "incorta_sqlx_host": headers.get("incorta-sqlx-host") or os.getenv("INCORTA_SQLX_HOST"),

            # Qdrant credentials
            "qdrant_url": headers.get("qdrant-url") or os.getenv("QDRANT_URL"),
            "qdrant_api_key": headers.get("qdrant-api-key") or os.getenv("QDRANT_API_KEY"),

            # Confluence credentials
            "confluence_url": headers.get("confluence-url") or os.getenv("CONFLUENCE_URL"),
            "confluence_token": headers.get("confluence-token") or os.getenv("CONFLUENCE_API_TOKEN"),

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
        """Start Slack OAuth flow."""
        client_id = os.getenv("SLACK_CLIENT_ID")
        redirect_uri = os.getenv("SLACK_REDIRECT_URI", f"http://localhost:{port}/auth/callback")

        if not client_id:
            return HTMLResponse(
                "<h1>Error</h1><p>SLACK_CLIENT_ID not configured</p>",
                status_code=500
            )

        scopes = [
            "channels:history", "channels:read",
            "groups:history", "groups:read",
            "search:read", "users:read",
        ]

        auth_url = (
            f"https://slack.com/oauth/v2/authorize?"
            f"client_id={client_id}&"
            f"user_scope={','.join(scopes)}&"
            f"redirect_uri={redirect_uri}"
        )

        return RedirectResponse(auth_url)

    async def handle_auth_callback(request: StarletteRequest):
        """OAuth callback - exchanges code for token and creates session."""
        code = request.query_params.get("code")
        error = request.query_params.get("error")

        if error:
            return HTMLResponse(f"<h1>Authentication Failed</h1><p>Error: {error}</p>", status_code=400)

        if not code:
            return HTMLResponse("<h1>Authentication Failed</h1><p>No authorization code</p>", status_code=400)

        # Exchange code for token
        client_id = os.getenv("SLACK_CLIENT_ID")
        client_secret = os.getenv("SLACK_CLIENT_SECRET")
        redirect_uri = os.getenv("SLACK_REDIRECT_URI", f"http://localhost:{port}/auth/callback")

        if not client_id or not client_secret:
            return HTMLResponse("<h1>Server Error</h1><p>OAuth credentials not configured</p>", status_code=500)

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
                return HTMLResponse(f"<h1>OAuth Failed</h1><p>Error: {error_msg}</p>", status_code=400)

            # Extract user token and create session
            slack_token = data["authed_user"]["access_token"]
            client = WebClient(token=slack_token)
            user_info = client.users_identity()
            user_name = user_info["user"]["name"]
            user_email = user_info["user"].get("email", "")

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
                        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                               max-width: 600px; margin: 50px auto; padding: 20px; background: #f5f5f5; }}
                        .container {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                        h1 {{ color: #2ecc71; }}
                        .session-id {{ background: #f8f9fa; padding: 15px; border-radius: 4px; 
                                       border-left: 4px solid #2ecc71; font-family: monospace; word-break: break-all; margin: 20px 0; }}
                        .copy-btn {{ background: #4A154B; color: white; border: none; padding: 10px 20px; 
                                     border-radius: 4px; cursor: pointer; }}
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
                        <p><small>Session expires after 30 days of inactivity.</small></p>
                    </div>
                    <script>
                        function copySessionId() {{
                            navigator.clipboard.writeText(document.getElementById('sessionId').textContent);
                            document.querySelector('.copy-btn').textContent = '✓ Copied!';
                            setTimeout(() => {{ document.querySelector('.copy-btn').textContent = 'Copy Session ID'; }}, 2000);
                        }}
                    </script>
                </body>
                </html>
            """)

        except Exception as e:
            logger.exception(f"Error during OAuth callback: {e}")
            return HTMLResponse(f"<h1>Error</h1><p>Failed to complete authentication: {str(e)}</p>", status_code=500)

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
            logger.info("Ibn Battouta MCP Server started!")
            logger.info(f"OAuth authentication: http://localhost:{port}/auth/slack")
            try:
                yield
            finally:
                logger.info("Shutting down...")

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

    logger.info(f"Starting on port {port}")
    logger.info(f"Available tools: {', '.join(get_all_tool_names())}")

    import uvicorn
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)
    return 0


if __name__ == "__main__":
    main()
