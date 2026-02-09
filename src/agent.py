import os
import json
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

from tools.qdrant_tool import search_knowledge_base as _search_knowledge_base

SYSTEM_PROMPT = """You are an expert Incorta assistant with access to comprehensive Incorta documentation through the search_knowledge_base tool.

**Your Knowledge Base includes:**
- Incorta Community Documentation
- Official Incorta Documentation
- Incorta Support Documentation

**Your Capabilities:**
1. Answer questions about Incorta features, functionality, and best practices
2. Provide guidance on Incorta configuration and administration
3. Help troubleshoot Incorta-related issues
4. Explain Incorta concepts

**Instructions:**
- ALWAYS search the knowledge base before answering questions
- Cite sources with URLs when available
- If no relevant results are found, say so clearly
- Be concise but thorough in your responses
"""


@tool
def search_knowledge_base(query: str, limit: int = 5) -> str:
    """Search the knowledge base using vector similarity. Contains Incorta Community articles, official documentation, and support articles.
    Best for: product features, official documentation, authoritative product information.
    Use this for any Incorta product-related queries.
    Returns: Article titles, URLs, text excerpts, relevance scores with source='knowledge_base'."""
    result = _search_knowledge_base({"query": query, "limit": limit})
    return json.dumps(result, default=str)


tools = [search_knowledge_base]

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0.7,
    max_tokens=8192,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

agent = create_react_agent(llm, tools)


def get_langfuse_handler():
    return LangfuseCallbackHandler()


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
