"""
GitHub Intelligence Agent

An async LangGraph agent that connects to GitHub via MCP (Multi-Server MCP Client)
and uses ChatOllama Cloud (kimi-k2.6) with an async SQLite checkpointer.

Usage:
    uv run python agent.py "Summarize open PRs for langchain-ai/langchain"

The agent supports:
- Multi-server MCP client connections
- Tool-calling loop (ReAct pattern) via LangGraph
- Persistent conversation state via AsyncSqliteSaver
- Streaming and non-streaming execution modes
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, Literal

from dotenv import load_dotenv

# Load environment variables before any other imports
load_dotenv()

from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient


def _require_env(key: str) -> str:
    """Return *key* from the environment or raise RuntimeError."""
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GITHUB_PAT: str = _require_env("GITHUB_PAT")
OLLAMA_API_KEY: str = _require_env("OLLAMA_API_KEY")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "kimi-k2.6:cloud")
DB_PATH: str = os.getenv("DB_PATH", ".checkpoints/checkpoints.db")
MCP_SERVER_NAME: str = "github"
MCP_SERVER_URL: str = "https://api.githubcopilot.com/mcp/"
SYSTEM_PROMPT: str = (
    "You are an expert GitHub Intelligence Agent. "
    "You have access to GitHub tools via MCP. "
    "Use the tools to fetch repository data, analyze PRs, issues, commits, and contributors. "
    "Always provide concise, actionable insights. "
    "When you don't know something, use the tools to find out rather than guessing."
)
DEFAULT_QUERY: str = (
    "Analyze the langchain-ai/langchain repository. "
    "List the 5 most recent open pull requests and summarize what each one does."
)


def _build_mcp_config() -> dict[str, Any]:
    """Build the MCP server configuration dict for MultiServerMCPClient."""
    # The raw config structure from the original agent.py is adapted here:
    #  - "type": "http" is mapped to "transport": "http"
    #  - headers.Authorization is populated with the GITHUB_PAT
    return {
        MCP_SERVER_NAME: {
            "transport": "http",
            "url": MCP_SERVER_URL,
            "headers": {"Authorization": f"Bearer {GITHUB_PAT}"} if GITHUB_PAT else {},
        }
    }


def _build_llm() -> ChatOllama:
    """
    Instantiate ChatOllama pointing at the Ollama Cloud endpoint.

    ``langchain-ollama`` exposes ``base_url`` (not ``host``) to configure the
    server address, and ``client_kwargs`` (not ``async_client_kwargs``) for
    extra ``httpx`` client options (e.g. auth headers).
    """
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.0,
        base_url=OLLAMA_BASE_URL,
        client_kwargs={
            "headers": {
                "Authorization": f"Bearer {OLLAMA_API_KEY}",
            }
        },
    )


# ---------------------------------------------------------------------------
# LangGraph nodes & edges
# ---------------------------------------------------------------------------

def should_continue(state: MessagesState) -> Literal["tools", END]:
    """Route to the tool node if the last assistant message contains tool_calls."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_graph(llm_with_tools: ChatOllama, tools: list[Any]) -> StateGraph:
    """Compile the ReAct-style LangGraph."""
    builder = StateGraph(MessagesState)

    # Define the async LLM node inline so LangGraph detects it as a coroutine function.
    async def _agent(state: MessagesState) -> dict[str, Any]:
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    builder.add_node("agent", _agent)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    return builder


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

async def run_agent(
    query: str,
    thread_id: str = "default",
    system_prompt: str = SYSTEM_PROMPT,
    stream: bool = False,
) -> str:
    """
    Execute the GitHub Intelligence Agent for a single user query.

    Parameters
    ----------
    query : str
        Natural-language instruction for the agent.
    thread_id : str, optional
        Conversation thread identifier (used by the SQLite checkpointer).
    system_prompt : str, optional
        System message prepended to the conversation.
    stream : bool, optional
        If *True*, stream tokens and only return the final assistant content.

    Returns
    -------
    str
        The final assistant response content.
    """
    # Ensure the checkpoint directory exists
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)

    mcp_config = _build_mcp_config()
    llm = _build_llm()

    # MultiServerMCPClient (v0.2.2+) is NOT an async context manager.
    # It is a lightweight config holder; get_tools() spawns and cleans up
    # per-server sessions internally on each call.
    mcp_client = MultiServerMCPClient(mcp_config)
    tools = await mcp_client.get_tools()
    llm_with_tools = llm.bind_tools(tools)

    # The sqlite checkpointer MUST be used as an async context manager.
    # from_conn_string returns an async generator, not a raw instance.
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        builder = build_graph(llm_with_tools, tools)
        graph = builder.compile(checkpointer=checkpointer)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ]
        config = {"configurable": {"thread_id": thread_id}}

        if stream:
            final_content = ""
            async for chunk in graph.astream(
                {"messages": messages},
                config=config,
                stream_mode="values",
            ):
                last_msg = chunk["messages"][-1]
                if hasattr(last_msg, "content"):
                    final_content = last_msg.content
            return str(final_content)

        result = await graph.ainvoke({"messages": messages}, config=config)
        last_message = result["messages"][-1]
        return str(getattr(last_message, "content", last_message))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for the GitHub Intelligence Agent."""
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    print(f"[GitHub Agent] Query: {query}\n")

    try:
        result = asyncio.run(run_agent(query, thread_id="cli_session_1"))
    except Exception as exc:
        print(f"[GitHub Agent] Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[GitHub Agent] Result:\n{result}")


if __name__ == "__main__":
    main()
