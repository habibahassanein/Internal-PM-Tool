"""
Product Manager Agent for LangChain ReAct Agent Executor.

Provides agent tools and setup for searching across multiple knowledge sources
including Confluence, Slack, Docs, Zendesk, and Jira.
"""

from typing import List, Optional
import logging
import os
import requests
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

from ..handler.confluence_handler import search_confluence, search_confluence_optimized as confluence_optimized_handler
from ..handler.slack_handler import search_slack_simplified
from ..handler.intent_analyzer import analyze_user_intent
from ..storage.cache_manager import get_cached_search_results, cache_search_results

logger = logging.getLogger(__name__)


def _get_secret_or_env(name: str, default: str = "") -> str:
    """
    Get value from Streamlit secrets first, then fall back to environment variables.
    
    Args:
        name: Name of the secret/environment variable
        default: Default value if not found
    
    Returns:
        Value from Streamlit secrets or environment variable
    """
    # Try Streamlit secrets first (if available and in Streamlit context)
    try:
        import streamlit as st
        # Check if we're in a Streamlit context and if the secret exists
        if hasattr(st, 'secrets') and name in st.secrets:
            return st.secrets.get(name, default)
    except Exception:
        # Streamlit not available, not in Streamlit context, or any error
        # Fall through to environment variables
        pass
    
    # Fall back to environment variables (loaded from .env by load_dotenv())
    return os.getenv(name, default)


# =========================
# Agent Tools
# =========================

@tool(
    "search_confluence_pages",
    description="""Search for pages in Confluence with AI-powered intent analysis.

    Args:
        query: Search query (natural language question)
        max_results: Maximum number of results to return (default 10)

    Returns:
        List of page dictionaries with metadata (title, url, excerpt, space, score)"""
)
def search_confluence_tool(
    query: str,
    max_results: int = 10
) -> List[dict]:
    """Search Confluence with intent analysis for better relevance."""
    if not query or not query.strip():
        return []

    try:
        # Analyze user intent to extract keywords, spaces, etc.
        intent_data = analyze_user_intent(query)

        logger.info(f"Confluence intent analysis: {intent_data.get('confluence_params', {})}")

        # Use the optimized handler with intent data
        results = confluence_optimized_handler(intent_data, query)

        # Limit results
        return results[:max_results]

    except Exception as e:
        logger.error(f"Confluence search failed: {e}")
        # Fallback to basic search
        return search_confluence(query, max_results, None)


@tool(
    "search_slack_messages",
    description="""Search for messages in Slack with AI-powered channel targeting.

    Args:
        query: Search query (natural language question)
        max_results: Maximum number of results to return (default 15)

    Returns:
        List of message dictionaries with enriched metadata (text, username, channel,
        date, permalink, score, thread_context, reactions)"""
)
def search_slack_tool(
    query: str,
    max_results: int = 15
) -> List[dict]:
    """Search Slack messages with intent analysis and channel intelligence."""
    if not query or not query.strip():
        return []

    try:
        # Analyze user intent to extract keywords, channels, time ranges, etc.
        intent_data = analyze_user_intent(query)

        logger.info(f"Slack intent analysis: {intent_data.get('slack_params', {})}")

        # Use the enhanced search with proper intent data
        results = search_slack_simplified(query, intent_data, max_results)

        # Return enriched results directly (no data stripping!)
        # This preserves: thread_context, reactions, scores, etc.
        return results

    except Exception as e:
        logger.error(f"Slack search failed: {e}")
        return []


@tool(
    "search_docs",
    description="""Search documents in Qdrant collection (Incorta Community, Docs & Support).
    
    Args:
        query: Search query
        limit: Number of results to return
    
    Returns:
        List of search results with metadata"""
)
def search_docs(query: str, limit: int = 5) -> List[dict]:
    """Search documents in Qdrant collection."""
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    
    if not query or not query.strip():
        return []
    
    try:
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
        
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
        query_vector = embedding_model.encode([query])[0]

        search_result = client.search(
            collection_name="docs",
            query_vector=("content_vector", query_vector),
            limit=limit,
            with_payload=True
        )
        
        # Format results
        formatted_results = []
        for r in search_result:
            if r.score >= 0.25:  # Minimum cosine score threshold
                formatted_results.append({
                    "title": r.payload.get("title", "") or "",
                    "url": r.payload.get("url", "") or "",
                    "text": r.payload.get("text", "") or "",
                    "score": r.score,
                    "source": "knowledge_base"
                })
        
        return formatted_results
    except Exception as e:
        logger.error(f"Failed to search docs: {e}")
        return []


@tool(
    "fetch_schema_details",
    description="""Fetch the details of a schema from Incorta.
    
    Args:
        schema_name: Name of the schema to fetch details for. (Only ZendeskTickets, Jira_F are supported)
    
    Returns:
        Details of the schema including tables and columns."""
)
def fetch_schema_details(schema_name: str) -> dict:
    """Fetch schema details from the Incorta environment."""
    env_url = os.getenv("INCORTA_ENV_URL")
    tenant = os.getenv("INCORTA_TENANT")
    user = os.getenv("INCORTA_USERNAME")  # Fix: use os.getenv instead of undefined variable
    password = os.getenv("INCORTA_PASSWORD")
    
    if not all([env_url, tenant, user, password]):
        return {"error": "Missing Incorta credentials. Set INCORTA_ENV_URL, INCORTA_TENANT, INCORTA_USERNAME, INCORTA_PASSWORD"}
    
    try:
        # Login to get session
        response = requests.post(
            f"{env_url}/authservice/login",
            data={"tenant": tenant, "user": user, "pass": password},
            verify=True,
            timeout=60
        )
        
        if response.status_code != 200:
            return {"error": f"Login failed: {response.status_code}"}
        
        # Extract session cookies
        id_cookie, login_id = None, None
        for item in response.cookies.items():
            if item[0].startswith("JSESSIONID"):
                id_cookie, login_id = item
                break
        
        if not id_cookie or not login_id:
            return {"error": "Failed to retrieve session cookies"}
        
        # Verify login and get CSRF token
        response = requests.get(
            f"{env_url}/service/user/isLoggedIn",
            cookies={id_cookie: login_id},
            verify=True,
            timeout=60
        )
        
        if response.status_code != 200 or "XSRF-TOKEN" not in response.cookies:
            return {"error": "Failed to verify login"}
        
        csrf_token = response.cookies["XSRF-TOKEN"]
        authorization = response.json().get("accessToken")
        
        if not authorization:
            return {"error": "Failed to retrieve access token"}
        
        # Fetch schema details
        url = f"{env_url}/bff/v1/schemas/name/{schema_name}"
        cookie = f"{id_cookie}={login_id};XSRF-TOKEN={csrf_token}"
        
        headers = {
            "Authorization": f"Bearer {authorization}",
            "Content-Type": "application/json",
            "X-XSRF-TOKEN": csrf_token,
            "Cookie": cookie
        }
        
        response = requests.get(url, headers=headers, verify=True, timeout=60)
        
        if response.status_code == 200:
            return {"schema_details": response.json()}
        else:
            return {"error": f"Failed to fetch schema details: {response.status_code} - {response.text}"}
    
    except Exception as e:
        logger.error(f"Failed to fetch schema details: {e}")
        return {"error": f"Exception: {str(e)}"}


@tool(
    "fetch_table_data",
    description="""Fetch data from a specified table in the schema using SQL query.
    
    Args:
        spark_sql: SQL query to fetch data from the table.
    
    Returns:
        Data from the table including columns and rows."""
)
def fetch_table_data(spark_sql: str) -> dict:
    """Fetch table data from the Incorta environment."""
    env_url = os.getenv("INCORTA_ENV_URL")
    tenant = os.getenv("INCORTA_TENANT")
    user = os.getenv("INCORTA_USERNAME")  # Fix: use os.getenv instead of undefined variable
    password = os.getenv("INCORTA_PASSWORD")
    
    if not all([env_url, tenant, user, password]):
        return {"error": "Missing Incorta credentials"}
    
    try:
        # Login to get session
        response = requests.post(
            f"{env_url}/authservice/login",
            data={"tenant": tenant, "user": user, "pass": password},
            verify=True,
            timeout=60
        )
        
        if response.status_code != 200:
            return {"error": f"Login failed: {response.status_code}"}
        
        # Extract session cookies
        id_cookie, login_id = None, None
        for item in response.cookies.items():
            if item[0].startswith("JSESSIONID"):
                id_cookie, login_id = item
                break
        
        if not id_cookie or not login_id:
            return {"error": "Failed to retrieve session cookies"}
        
        # Verify login and get CSRF token
        response = requests.get(
            f"{env_url}/service/user/isLoggedIn",
            cookies={id_cookie: login_id},
            verify=True,
            timeout=60
        )
        
        if response.status_code != 200 or "XSRF-TOKEN" not in response.cookies:
            return {"error": "Failed to verify login"}
        
        csrf_token = response.cookies["XSRF-TOKEN"]
        authorization = response.json().get("accessToken")
        
        if not authorization:
            return {"error": "Failed to retrieve access token"}
        
        # Execute SQL query
        url = f"{env_url}/bff/v1/sqlxquery"
        cookie = f"{id_cookie}={login_id};XSRF-TOKEN={csrf_token}"
        
        headers = {
            "Authorization": f"Bearer {authorization}",
            "Content-Type": "application/json",
            "X-XSRF-TOKEN": csrf_token,
            "Cookie": cookie
        }
        
        params = {"sql": spark_sql}
        response = requests.post(url, headers=headers, json=params, verify=True, timeout=60)
        
        if response.status_code == 200:
            return {"data": response.json()}
        else:
            return {"error": f"Failed to fetch data: {response.status_code} - {response.text}"}
    
    except Exception as e:
        logger.error(f"Failed to fetch table data: {e}")
        return {"error": f"Exception: {str(e)}"}


# =========================
# Agent Setup
# =========================

class RetryAgentExecutor:
    """
    Wrapper for AgentExecutor that automatically retries with different API keys on quota errors.
    """

    def __init__(self, api_manager, tools, prompt, max_retries: int = 4):
        """
        Initialize retry agent executor.

        Args:
            api_manager: GeminiAPIManager instance
            tools: List of agent tools
            prompt: Agent prompt template
            max_retries: Maximum number of retries (defaults to number of API keys)
        """
        self.api_manager = api_manager
        self.tools = tools
        self.prompt = prompt
        self.max_retries = min(max_retries, len(api_manager.api_keys))
        self.current_executor = None
        self._create_executor()

    def _create_executor(self):
        """Create a new agent executor with current API key."""
        llm = self.api_manager.get_llm()
        agent = create_react_agent(llm, self.tools, self.prompt)
        self.current_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            handle_parsing_errors=True,
            verbose=True,
            max_iterations=10
        )

    def stream(self, inputs: dict):
        """
        Stream agent execution with automatic retry on quota errors.

        Args:
            inputs: Agent inputs (e.g., {"input": "query"})

        Yields:
            Agent execution chunks
        """
        attempt = 0
        last_error = None

        while attempt < self.max_retries:
            try:
                # Stream from current executor
                for chunk in self.current_executor.stream(inputs):
                    yield chunk

                # If we get here, execution succeeded
                self.api_manager.mark_success()
                return

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Check if it's a quota/rate limit error
                if any(keyword in error_str for keyword in ["quota", "429", "rate limit", "resourceexhausted"]):
                    attempt += 1
                    logger.warning(f"Quota error on attempt {attempt}/{self.max_retries}. Rotating API key...")

                    if attempt < self.max_retries:
                        # Mark failure and rotate key
                        self.api_manager.mark_failure(error_type="quota")

                        # Recreate executor with new API key
                        self._create_executor()

                        # Small delay before retry
                        import time
                        time.sleep(1)
                        continue
                    else:
                        # All keys exhausted
                        raise Exception(f"All {self.max_retries} API keys exhausted quota limits. Please wait or upgrade your plan.") from e
                else:
                    # Non-quota error, re-raise immediately
                    logger.error(f"Non-quota error: {e}")
                    raise

        # If we exit the loop, all retries failed
        raise Exception(f"Failed after {self.max_retries} attempts") from last_error


def create_pm_agent(api_key: Optional[str] = None):
    """
    Create and configure the Product Manager agent with API key rotation support.

    Args:
        api_key: Optional Gemini API key (uses API manager with multiple keys if not provided)

    Returns:
        Configured RetryAgentExecutor or AgentExecutor instance
    """
    # Try to use API manager for multiple keys
    api_manager = None
    if not api_key:
        try:
            from ..api_manager import create_api_manager_from_env
            api_manager = create_api_manager_from_env()
            logger.info(f"Agent: Using API manager with {len(api_manager.api_keys)} key(s)")
        except Exception as e:
            logger.warning(f"Agent: Failed to create API manager: {e}, using single key")
            api_key = _get_secret_or_env("GEMINI_API_KEY")

    if not api_manager and not api_key:
        raise ValueError("GEMINI_API_KEY must be set in environment, Streamlit secrets, or provided as argument")

    # Define tools (using refactored versions with intent analysis)
    tools = [
        search_confluence_tool,  # Updated: now uses intent analysis
        search_slack_tool,        # Updated: now uses intent analysis + channel intelligence
        search_docs,
        fetch_schema_details,
        fetch_table_data
    ]

    # Agent prompt template
    template = """You are an AI Assistant that helps Product Managers search across multiple internal knowledge bases including Confluence, Slack, Docs, Zendesk, and Jira.

Your available tools are:
{tools}

IMPORTANT SEARCH GUIDELINES:

1. Use `search_confluence_pages` to find Confluence documentation:
   - The tool uses AI to automatically target relevant spaces
   - Just pass the user's question directly - no need to extract keywords

2. Use `search_slack_messages` to find Slack conversations:
   - The tool uses channel intelligence to search the most relevant channels
   - Returns enriched results with thread context and engagement metrics
   - Just pass the user's question directly

3. Use `search_docs` to search the knowledge base (Incorta Community, Docs & Support)

4. Use `fetch_schema_details` for Zendesk and Jira database schemas:
   - Input must be EXACTLY "ZendeskTickets" for Zendesk questions
   - Input must be EXACTLY "Jira_F" for Jira questions

5. Use `fetch_table_data` to query data from Zendesk/Jira tables

SEARCH STRATEGY:
- By default, search ALL relevant sources (Confluence, Slack, Docs) for comprehensive answers
- If the user specifically asks for one source (e.g., "search Slack"), use only that tool
- Each tool automatically handles relevance scoring - trust the results
- If you get no results, try rephrasing the query with synonyms or related terms

RESPONSE FORMAT:
- Provide clear, actionable answers based on the search results
- Include recommendations from Zendesk/Jira tickets when relevant
- Cite specific sources in your answer (channel names, page titles, etc.)

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)

    # If we have API manager, use retry wrapper
    if api_manager:
        logger.info("Creating RetryAgentExecutor with automatic API key rotation")
        return RetryAgentExecutor(api_manager, tools, prompt)

    # Otherwise, create standard executor with single key
    logger.info("Creating standard AgentExecutor with single API key")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.1,
        api_key=api_key
    )

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=10
    )

    return agent_executor


__all__ = [
    "create_pm_agent",
    "search_confluence_optimized",
    "search_slack_messages",
    "search_docs",
    "fetch_schema_details",
    "fetch_table_data"
]

