from fastmcp import FastMCP
from fastmcp.server.dependencies import get_context
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import JSONResponse, HTMLResponse

from typing import Optional

from auth import context
from auth.oauth_config import get_oauth_config, reload_oauth_config
from auth.oauth_handler import exchange_code_for_token
from auth.session_middleware import SlackSessionMiddleware

from context.user_context import user_context
from tools.search_community_tool import fetch_community_tool
from tools.search_docs_tool import fetch_docs_tool, DocVersion
from tools.search_support_tool import fetch_support_tool
from dotenv import load_dotenv

load_dotenv()


import logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("uvicorn.error")

PORT = 8080

reload_oauth_config()

session_middleware = Middleware(SlackSessionMiddleware)

class SecureFastMCP(FastMCP):
    """Custom FastMCP with session middleware for secure authentication."""
    def __init__(self, name: str):
        super().__init__(
            name=name,
            host="0.0.0.0",
            port=PORT,
            stateless_http=False
        )

    def streamable_http_app(self):
        """Override to add secure middleware stack."""
        app = super().streamable_http_app()

        # Add session middleware
        app.user_middleware.insert(0, session_middleware)

        # Rebuild middleware stack
        app.middleware_stack = app.build_middleware_stack()
        logger.info("Added SlackSessionMiddleware for secure authentication")
        return app


mcp = SecureFastMCP("Internal PM Tool MCP Server")


@mcp.tool(
    "search_community",
    description=(
        "Search Incorta Community forums for user discussions, solutions, and tips. "
        "Best for: user experiences, community solutions, practical tips. "
        "Returns: Post titles, URLs, excerpts with source='community'. "
        "\n\n**USAGE RULES:**\n"
        "- Use for: User experiences, community solutions, practical tips.\n"
        "- Source Priority: HIGH for troubleshooting & practical tips.\n"
        "- Citation: Always include source='community' and quote key evidence from results."
    )
)
def tool_search_community(query: str, max_results: int = 5):
    return fetch_community_tool(query=query, max_results=max_results)


@mcp.tool(
    "search_docs",
    description=(
        "Search Incorta official documentation for product features, setup guides, and technical details. "
        "Best for: product features, official documentation, setup instructions. "
        "Returns: Document titles, URLs, excerpts with source='docs'. "
        "\n\n**USAGE RULES:**\n"
        "- Use for: Official product docs, setup guides, technical details.\n"
        "- Source Priority: HIGHEST for product features & documentation.\n"
        "- Citation: Always include source='docs' and quote key evidence from results."
        "- Versioning: Specify document version when relevant to ensure accurate info. (Available versions: latest, cloud, 6.0, 5.2, 5.1)"
    )
)
def tool_search_docs(query: str, max_results: int = 5, version: DocVersion = DocVersion.LATEST.value):
    return fetch_docs_tool(query=query, max_results=max_results, version=version)


@mcp.tool(
    "search_support",
    description=(
        "Search Incorta Support articles for troubleshooting steps, known issues, and resolutions. "
        "Best for: troubleshooting, known issues, error resolutions. "
        "Returns: Article titles, URLs, excerpts with source='support'. "
        "\n\n**USAGE RULES:**\n"
        "- Use for: Troubleshooting, known issues, error resolutions.\n"
        "- Source Priority: HIGHEST for troubleshooting & error resolutions.\n"
        "- Citation: Always include source='support' and quote key evidence from results."
    )
)
def tool_search_support(query: str, max_results: int = 5):
    return fetch_support_tool(query=query, max_results=max_results)


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: StarletteRequest):
    """Health check endpoint for load balancer."""
    return JSONResponse({"status": "healthy"})



@mcp.custom_route("/oauth2callback", methods=["GET"])
async def oauth_callback(request: StarletteRequest) -> HTMLResponse:
    """
    Handle OAuth callback from Slack.

    This endpoint exchanges the authorization code for a token and binds
    it to the current session for secure access.
    """
    code = request.query_params.get("code")
    error = request.query_params.get("error")

    if error:
        return HTMLResponse(
            content=f"""
            <html>
                <body>
                    <h1>OAuth Error</h1>
                    <p>Error: {error}</p>
                    <p>You can close this window.</p>
                </body>
            </html>
            """,
            status_code=400,
        )

    if not code:
        return HTMLResponse(
            content="""
            <html>
                <body>
                    <h1>OAuth Error</h1>
                    <p>No authorization code received.</p>
                    <p>You can close this window.</p>
                </body>
            </html>
            """,
            status_code=400,
        )

    # Extract state parameter for CSRF validation
    state = request.query_params.get("state")

    if not state:
        return HTMLResponse(
            content="""
            <html>
                <body>
                    <h1>Authentication Failed</h1>
                    <p>Error: Missing OAuth state parameter.</p>
                    <p>This may indicate a CSRF attack attempt.</p>
                    <p>You can close this window.</p>
                </body>
            </html>
            """,
            status_code=400,
        )

    # Validate OAuth state parameter and get the bound session ID (CSRF protection)
    from auth.session_store import get_session_store

    store = get_session_store()

    # Get the session_id that was bound to this state
    # We need to look it up before validation to know which session to validate against
    with store._lock:
        if state not in store._oauth_states:
            logger.error(f"Invalid OAuth state: {state} not found")
            return HTMLResponse(
                content="""
                <html>
                    <body>
                        <h1>Authentication Failed</h1>
                        <p>Error: Invalid or expired OAuth state parameter.</p>
                        <p>Please generate a new OAuth URL using the slack_get_oauth_url tool.</p>
                        <p>You can close this window.</p>
                    </body>
                </html>
                """,
                status_code=400,
            )
        session_id, _ = store._oauth_states[state]

    logger.info(f"OAuth callback with session ID: {session_id}")

    if not store.validate_and_consume_oauth_state(state, session_id):
        logger.error(f"SECURITY: Invalid OAuth state for session {session_id}")
        return HTMLResponse(
            content="""
            <html>
                <body>
                    <h1>Authentication Failed</h1>
                    <p>Error: Invalid or expired OAuth state parameter.</p>
                    <p>This may indicate a CSRF attack attempt or an expired authorization request.</p>
                    <p>Please generate a new OAuth URL and try again.</p>
                    <p>You can close this window.</p>
                </body>
            </html>
            """,
            status_code=400,
        )

    context.fastmcp_session_id.set(session_id)

    try:
        # Exchange code for token (will bind to session automatically)
        token, user_id, exchange_error = await exchange_code_for_token(code)

        user_context.set({
            "slack_token": token,
            "session_id": session_id,
        })

        if exchange_error:
            return HTMLResponse(
                content=f"""
                <html>
                    <body>
                        <h1>Authentication Failed</h1>
                        <p>Error: {exchange_error}</p>
                        <p>You can close this window.</p>
                    </body>
                </html>
                """,
                status_code=500,
            )

        return HTMLResponse(
            content=f"""
            <html>
                <body>
                    <h1>✅ Authentication Successful!</h1>
                    <p>You have been authenticated as user: <strong>{user_id}</strong></p>
                    <p>Your session is now authorized to access Slack.</p>
                    <p>You can close this window and return to Cursor.</p>
                </body>
            </html>
            """
        )
    except Exception as e:
        # Catch any unexpected exceptions and return user-friendly error
        logger.error(f"Unexpected error in OAuth callback: {e}", exc_info=True)
        return HTMLResponse(
            content=f"""
            <html>
                <body>
                    <h1>Authentication Failed</h1>
                    <p>An unexpected error occurred during authentication.</p>
                    <p>Error: {e!s}</p>
                    <p>You can close this window and try again.</p>
                </body>
            </html>
            """,
            status_code=500,
        )


app = mcp.http_app(
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ]
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        log_level="debug",
        workers=1, 
        proxy_headers=True,
        forwarded_allow_ips="*"
    )
