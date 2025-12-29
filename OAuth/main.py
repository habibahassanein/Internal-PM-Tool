from fastmcp import FastMCP
# from fastapi import Request
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
from tools.confluence_tool import search_confluence
from tools.slack_tool import search_slack
# from tools.qdrant_tool import search_knowledge_base
from tools.search_community_tool import fetch_community_tool
from tools.search_docs_tool import fetch_docs_tool, DocVersion
from tools.search_support_tool import fetch_support_tool
from tools.incorta_tools import query_zendesk, query_jira, get_zendesk_schema, get_jira_schema
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
    "search_confluence",
    description=(
        "Search Confluence pages for internal documentation, project pages, and process guides. "
        "Best for: internal processes, best practices, detailed project documentation. "
        "Returns: Page titles, URLs, excerpts with source='confluence'. "
        "\n\n**USAGE RULES:**\n"
        "- **MANDATORY BASELINE**: Run for EVERY query before producing an answer.\n"
        "- Source Priority: HIGHEST for internal processes & best practices, HIGH for product features.\n"
        "- Citation: Always include source='confluence' and quote key evidence from results.\n"
        "- Multi-Source Synthesis: Cross-reference with docs, community, support, slack. When sources conflict, cite both with dates."
    )
)
def tool_search_confluence(query: str, max_results: int = 10, space_filter: Optional[str] = None):
    args = {
        "query": query,
        "max_results": max_results,
        "space_filter": space_filter
    }
    return search_confluence(args)


@mcp.tool(
    "search_slack",
    description=(
        "Search Slack messages for team communications, decisions, and quick updates. "
        "Best for: recent discussions, team decisions, quick updates, release announcements. "
        "Returns: Message excerpts, timestamps, user info with source='slack'. "
        "\n\n**USAGE RULES:**\n"
        "- **MANDATORY BASELINE**: Run for EVERY query before producing an answer (along with docs, community, support, confluence).\n"
        "- Source Priority: HIGHEST for release dates & announcements, HIGH for troubleshooting.\n"
        "- Citation: Include username/channel when relevant (e.g., 'According to @user in #release-announcements').\n"
        "- Authentication: Requires OAuth - use get_slack_oauth_url if not authenticated.\n"
        "- If no results: Mention 'No relevant Slack discussions found' in your answer.\n"
        "- Multi-Source Synthesis: Combine with community, docs, support and confluence for comprehensive answers."
    )
)
def tool_search_slack(query: str, max_results: int = 10, channel_filter: Optional[str] = None):
    from auth.session_store import get_session_store
    
    # Get session ID from FastMCP context and set it in our context variable
    try:
        ctx = get_context()
        if ctx and hasattr(ctx, "session_id"):
            session_id = ctx.session_id
            context.fastmcp_session_id.set(session_id)
            logger.info(f"search_slack: Set session ID: {session_id}")
            
            # Also lookup and set the authenticated user ID
            store = get_session_store()
            user_id = store.get_user_by_session(session_id)
            if user_id:
                context.authenticated_user_id.set(user_id)
                logger.info(f"search_slack: Set user ID: {user_id}")
            else:
                logger.warning(f"search_slack: No user bound to session {session_id}")
    except Exception as e:
        logger.error(f"search_slack: Error getting context: {e}")
    
    args = {
        "query": query,
        "max_results": max_results,
        "channel_filter": channel_filter
    }
    return search_slack(args)


@mcp.tool(
    "get_slack_oauth_url",
    description=(
        "Get the OAuth authorization URL for users to authenticate with Slack. "
        "\n\n**USAGE:** Call this tool when search_slack fails due to missing authentication. "
        "The user must visit the returned URL to authorize access before using search_slack. "
        "\n\n**IMPORTANT**: Since search_slack is MANDATORY for every query, if authentication fails, "
        "you MUST call this tool, get the auth URL, present it to the user, and wait for them to authenticate "
        "before proceeding with the search. "
        "\nReturns: Dictionary with authorization URL and instructions."
    )
)
def slack_get_oauth_url() -> dict:
    """
    Get the OAuth authorization URL for users to authenticate with Slack.
    
    **USAGE:** Call this tool when search_slack fails due to missing authentication.
    The user must visit the returned URL to authorize access before using search_slack.

    By default use `search_slack` first before calling this tool.

    Returns:
        Dictionary with authorization URL and instructions
    """
    from auth.session_store import get_session_store

    config = get_oauth_config()
    if not config.is_configured():
        return {
            "ok": False,
            "error": "OAuth not configured. Please set SLACK_CLIENT_ID and SLACK_CLIENT_SECRET.",
        }
    
    session_id = None
    try:
        ctx = get_context()
        if ctx and hasattr(ctx, "session_id"):
            session_id = ctx.session_id
            logger.info(f"Got FastMCP session ID: {session_id}")
            # Also set it in our context variable for other functions
            context.fastmcp_session_id.set(session_id)
    except Exception as e:
        logger.error(f"Error getting FastMCP context: {e}")

    if not session_id:
        return {
            "ok": False,
            "error": "No session ID found. Unable to generate OAuth URL.",
        }

    store = get_session_store()
    state = store.generate_oauth_state(session_id)

    return {
        "authorization_url": config.get_authorization_url(state=state),
        "instructions": (
            "Visit the authorization URL to authenticate. "
        ),
    }

@mcp.tool(
    "search_community",
    description=(
        "Search Incorta Community forums for user discussions, solutions, and tips. "
        "Best for: user experiences, community solutions, practical tips. "
        "Returns: Post titles, URLs, excerpts with source='community'. "
        "\n\n**USAGE RULES:**\n"
        "- **MANDATORY BASELINE**: Run for EVERY query before producing an answer.\n"
        "- Source Priority: HIGH for troubleshooting & practical tips.\n"
        "- Citation: Always include source='community' and quote key evidence. Note patterns if multiple users report same issue.\n"
        "- If no results: Mention 'No relevant community discussions found' in your answer."
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
        "- **MANDATORY BASELINE**: Run for EVERY query before producing an answer.\n"
        "- Source Priority: HIGHEST for product features & documentation.\n"
        "- Citation: Always include source='docs' and quote key evidence. Preserve version numbers, dates, IDs.\n"
        "- Versioning: Specify document version when relevant (Available: latest, cloud, 6.0, 5.2, 5.1).\n"
        "\n**UPGRADE QUERY WORKFLOW:**\n"
        "If query contains 'upgrade', 'migration', 'upgrade path':\n"
        "1. FIRST search 'Incorta Release Support Policy' to get version timeline ordered by RELEASE DATE (not version number)\n"
        "2. Build sequential path from current → target version (list ALL interim versions)\n"
        "3. For EACH version search: '[VERSION] release notes', '[VERSION] upgrade considerations', '[FROM] to [TO] upgrade'\n"
        "4. Collect ALL findings in chronological order. Flag critical transitions (Spark/Python/Zookeeper changes)\n"
        "5. Also search search_confluence for upgrade information"
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
        "- **MANDATORY BASELINE**: Run for EVERY query before producing an answer.\n"
        "- Source Priority: HIGHEST for troubleshooting & error resolutions.\n"
        "- Citation: Always include source='support' and quote key evidence. Preserve error codes, version numbers.\n"
        "- If no results: Mention 'No relevant support articles found' in your answer."
    )
)
def tool_search_support(query: str, max_results: int = 5):
    return fetch_support_tool(query=query, max_results=max_results)


# @mcp.tool(
#     "search_knowledge_base",
#     description=(
#         "Search the knowledge base using vector similarity. Contains Incorta Community articles, "
#         "official documentation, and support articles. "
#         "Best for: product features, official documentation, authoritative product information. "
#         "Returns: Article titles, URLs, text excerpts, relevance scores with source='knowledge_base'. "
#         "\n\n**USAGE RULES:**\n"
#         "- Run this tool for EVERY query before producing an answer (mandatory baseline search).\n"
#         "- Use for: Official product docs, community articles, support content, authoritative information.\n"
#         "- Source Priority: HIGHEST for product features & documentation, HIGH for troubleshooting.\n"
#         "- Citation: Always include source='knowledge_base', preserve technical details (versions, dates, IDs).\n"
#         "- Multi-Source Synthesis: Cross-reference with confluence and other sources when available.\n"
#         "- Upgrade Queries: For upgrade questions, search for 'Incorta Release Support Policy' and version-specific considerations."
#     )
# )
# def tool_search_knowledge_base(query: str, limit: int = 10):
#     args = {
#         "query": query,
#         "limit": limit
#     }
#     return search_knowledge_base(args)


@mcp.tool(
    "get_zendesk_schema",
    description=(
        "Get Zendesk schema details from Incorta (tables and columns). "
        "Call this before querying Zendesk data to understand available fields. "
        "Returns: Schema structure with table names and column definitions. "
        "\n\n**USAGE RULES:**\n"
        "- **PREREQUISITE**: MUST call this before using query_zendesk to understand available fields.\n"
        "- Use for: Understanding Zendesk data structure, planning queries."
    )
)
def tool_get_zendesk_schema():
    args = {}
    return get_zendesk_schema(args)


@mcp.tool(
    "query_zendesk",
    description=(
        "Execute SQL query on Zendesk data in Incorta. "
        "Best for: customer issues, support trends, pain point patterns. "
        "Must call get_zendesk_schema first to understand available fields. "
        "Returns: Query results with columns and rows, source='zendesk'. "
        "\n\n**USAGE RULES:**\n"
        "- **ONLY WHEN EXPLICITLY ASKED**: Only call when user explicitly asks about Zendesk or support issues.\n"
        "- **PREREQUISITE**: MUST call get_zendesk_schema first to understand available fields.\n"
        "- Source Priority: HIGHEST for customer issues & pain points.\n"
        "- Citation: Note patterns if multiple customers report same issue, include source='zendesk'.\n"
        "- PM Value: Identify patterns across customer tickets, connect pain points to features."
    )
)
def tool_query_zendesk(query: str):
    args = {
        "spark_sql": query
    }
    return query_zendesk(args)


@mcp.tool(
    "get_jira_schema",
    description=(
        "Get Jira schema details from Incorta (tables and columns). "
        "Call this before querying Jira data to understand available fields. "
        "Returns: Schema structure with table names and column definitions. "
        "\n\n**USAGE RULES:**\n"
        "- **PREREQUISITE**: MUST call this before using query_jira to understand available fields.\n"
        "- Use for: Understanding Jira data structure, planning queries."
    )
)
def tool_get_jira_schema():
    args = {}
    return get_jira_schema(args)


@mcp.tool(
    "query_jira",
    description=(
        "Execute SQL query on Jira data in Incorta. "
        "Best for: development status, roadmap, feature progress, bug tracking. "
        "Must call get_jira_schema first to understand available fields. "
        "Returns: Query results with columns and rows, source='jira'. "
        "\n\n**USAGE RULES:**\n"
        "- **ONLY WHEN EXPLICITLY ASKED**: Only call when user explicitly asks about Jira or engineering status.\n"
        "- **PREREQUISITE**: MUST call get_jira_schema first to understand available fields.\n"
        "- Source Priority: HIGHEST for development status & roadmap.\n"
        "- Citation: Include issue status/priority if relevant (e.g., 'Jira ticket PROD-123 is In Progress').\n"
        "- PM Value: Connect customer issues (Zendesk) to development work (Jira), identify blockers."
    )
)
def tool_query_jira(query: str):
    args = {
        "spark_sql": query
    }
    return query_jira(args)


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
