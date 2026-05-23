# GitHub Intelligence Agent

An **async LangGraph** agent that connects to GitHub via **MCP (Multi-Server MCP Client)** and uses **ChatOllama Cloud** (`kimi-k2.6`) with an **async SQLite checkpointer** for persistent conversation state.

## Features

- **LangGraph ReAct Agent** — agent-node → tool-node → agent-node loop with conditional edges
- **Multi-Server MCP Client** — connects to the GitHub MCP server over HTTP with Bearer auth
- **ChatOllama Cloud** — uses the Ollama Cloud API endpoint with API-key authentication (`kimi-k2.6:cloud`)
- **Async SQLite Checkpointer** — persistent conversation threads via `AsyncSqliteSaver`
- **Streaming support** — optional real-time token streaming via `graph.astream(..., stream_mode="values")`

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment

Create (or edit) `.env`:

```env
GITHUB_PAT="ghp_..."
OLLAMA_API_KEY="sk-..."
OLLAMA_BASE_URL="https://api.ollama.ai"
OLLAMA_MODEL="kimi-k2.6:cloud"
```

> A `.env` file is **already provided** in this repo.

### 3. Run the agent

```bash
# Default query
uv run python agent.py

# Custom query
uv run python agent.py "List open issues in langchain-ai/langchain and summarize them"
```

## Architecture

```
agent.py
├── Configuration (env vars, MCP config, LLM config)
├── build_graph() → StateGraph(MessagesState)
│   ├── agent node   → ChatOllama.ainvoke(state["messages"])
│   ├── tools node   → ToolNode(tools)  (parallel MCP tool execution)
│   ├── conditional  → should_continue(state) → "tools" | END
│   └── edge         → tools → agent (ReAct loop)
├── run_agent(query, thread_id) → str
│   ├── MultiServerMCPClient (async context manager)
│   ├── AsyncSqliteSaver (async context manager)
│   └── graph.ainvoke(...) | graph.astream(...)
└── main() → CLI entry point
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `langgraph` | Async state-graph builder + prebuilt `ToolNode` |
| `langchain-mcp-adapters` | `MultiServerMCPClient` for MCP HTTP connections |
| `langchain-ollama` | `ChatOllama` with cloud `host` + `async_client_kwargs` |
| `langgraph-checkpoint-sqlite` | `AsyncSqliteSaver` for persistent thread state |
| `python-dotenv` | Load `.env` variables |
| `aiohttp` | Async HTTP transport for MCP client |

## MCP Server Config

The original MCP config from the provided `agent.py` snippet:

```json
{
  "mcpServers": {
    "github": {
      "type": "http",
      "url": "https://api.githubcopilot.com/mcp/",
      "headers": { "Authorization": "{GITHUB_PAT}" }
    }
  }
}
```

...is adapted in code to the `langchain-mcp-adapters` format (`transport: http`, `headers: Bearer ...`).

## Extending

- Add more MCP servers to `_build_mcp_config()`
- Switch models by setting `OLLAMA_MODEL` env var
- Change checkpoint DB path via `DB_PATH`
- Enable streaming by passing `stream=True` to `run_agent()`

## License

MIT
