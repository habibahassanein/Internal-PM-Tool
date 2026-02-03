import os
import json
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

from tools.search_community_tool import fetch_community_tool
from tools.search_docs_tool import fetch_docs_tool, DocVersion
from tools.search_support_tool import fetch_support_tool

SYSTEM_PROMPT = """You are an expert Incorta assistant with access to comprehensive Incorta documentation through search tools.

**Your Knowledge Base includes:**
- Incorta Community Documentation (user discussions, solutions, tips)
- Official Incorta Documentation (product features, setup guides, technical details)
- Incorta Support Documentation (troubleshooting, known issues, resolutions)

**Your Capabilities:**
1. Answer questions about Incorta features, functionality, and best practices
2. Provide guidance on Incorta configuration and administration
3. Help troubleshoot Incorta-related issues
4. Explain Incorta concepts

**Instructions:**
- ALWAYS search the knowledge base before answering questions
- Use multiple search tools when the query spans different areas
- Cite sources with URLs when available
- If no relevant results are found, say so clearly
- Be concise but thorough in your responses
"""


@tool
def search_docs(query: str, max_results: int = 5) -> str:
    """Search Incorta official documentation for product features, setup guides, and technical details.
    Best for: product features, official documentation, setup instructions.
    Available versions: latest, cloud, 6.0, 5.2, 5.1"""
    result = fetch_docs_tool(query=query, max_results=max_results, version=DocVersion.LATEST.value)
    return json.dumps(result, default=str)


@tool
def search_community(query: str, max_results: int = 5) -> str:
    """Search Incorta Community forums for user discussions, solutions, and practical tips.
    Best for: user experiences, community solutions, practical tips."""
    result = fetch_community_tool(query=query, max_results=max_results)
    return json.dumps(result, default=str)


@tool
def search_support(query: str, max_results: int = 5) -> str:
    """Search Incorta Support articles for troubleshooting steps, known issues, and resolutions.
    Best for: troubleshooting, known issues, error resolutions."""
    result = fetch_support_tool(query=query, max_results=max_results)
    return json.dumps(result, default=str)


tools = [search_docs, search_community, search_support]

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0.7,
    max_tokens=8192,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

agent = create_react_agent(llm, tools)


def get_langfuse_handler():
    return LangfuseCallbackHandler(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        host=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
    )


async def chat(message: str) -> str:
    """Run the agent with a user message and return the final response."""
    handler = get_langfuse_handler()

    inputs = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=message),
        ]
    }

    result = await agent.ainvoke(inputs, config={"callbacks": [handler]})

    # Extract the last AI message
    ai_messages = [m for m in result["messages"] if m.type == "ai" and m.content]
    if ai_messages:
        last = ai_messages[-1]
        if isinstance(last.content, list):
            # Handle structured content blocks
            return "".join(block.get("text", "") for block in last.content if isinstance(block, dict))
        return last.content

    return "No response generated."
