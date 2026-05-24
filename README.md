# GitHub Intelligence Agent — User Manual

A production-ready **async LangGraph** agent that connects to **GitHub via MCP** (Multi-Server MCP Client) and answers natural-language queries using **ChatOllama Cloud** (`kimi-k2.6:cloud`) with persistent conversation memory powered by an **async SQLite checkpointer**.

<div align="center" style="font-size:25px;">
Demo :
<br>
<br>
<video src="[Screen Recording 2026-05-24 173111.mp4](https://github.com/Debvex/GitHub-Intelligence-Agent/blob/d1b3e74963573b2dfe696371ebcad53361d58353/Screen%20Recording%202026-05-24%20173111.mp4)" width="100%" controls></video>
</div>
---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [TUI Usage (`app.py`)](#tui-usage-apppy)
  - [Main Menu](#main-menu)
  - [Run Query](#1-run-query)
  - [Stream Query](#2-stream-query-live-tokens)
  - [Continue Thread](#3-continue-thread)
  - [View History](#4-view-history)
  - [Settings](#5-settings)
  - [Quit](#q-quit)
- [CLI Usage (`agent.py`)](#cli-usage-agentpy)
- [Architecture Overview](#architecture-overview)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)
- [Extending the Agent](#extending-the-agent)

---

## Quick Start

```bash
# 1. Clone & enter the project
cd github-intelligence-agent

# 2. Install dependencies (uv)
uv sync

# 3. Configure environment
cp .env.example .env   # or edit the existing .env
# Add your GITHUB_PAT + OLLAMA_API_KEY

# 4a. Launch the interactive TUI
uv run python app.py

# 4b. Or run a single query from the CLI
uv run python agent.py "Summarize open PRs for langchain-ai/langchain"
```

---

## Installation

Requires **Python ≥3.11** and **[uv](https://docs.astral.sh/uv/)**.

### Using `uv` (recommended)

```bash
uv sync
```

This installs all production dependencies listed in `pyproject.toml`:

| Package | Purpose |
|---------|---------|
| `langgraph` | Async state-graph builder + prebuilt `ToolNode` |
| `langchain-mcp-adapters` | `MultiServerMCPClient` for MCP HTTP connections |
| `langchain-ollama` | `ChatOllama` with cloud `host` + `async_client_kwargs` |
| `langgraph-checkpoint-sqlite` | `AsyncSqliteSaver` for persistent thread state |
| `python-dotenv` | Load `.env` variables |
| `aiohttp` | Async HTTP transport for MCP client |
| `rich` | Terminal UI panels, tables, spinners, live displays |

---

## Configuration

Create or edit `.env` in the project root:

```env
# ---- GitHub ----
GITHUB_PAT="ghp_xxxxxxxxxxxxxxxxxxxx"

# ---- Ollama Cloud ----
OLLAMA_API_KEY="sk-xxxxxxxxxxxxxxxxxxxx"
OLLAMA_BASE_URL="https://api.ollama.ai"
OLLAMA_MODEL="kimi-k2.6:cloud"

# ---- Optional ----
DB_PATH=".checkpoints/checkpoints.db"
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GITHUB_PAT` | ✅ | — | GitHub Personal Access Token with `repo` scope |
| `OLLAMA_API_KEY` | ✅ | — | Ollama Cloud API key |
| `OLLAMA_BASE_URL` | ❌ | `https://api.ollama.ai` | Ollama API endpoint |
| `OLLAMA_MODEL` | ❌ | `kimi-k2.6:cloud` | Model name on Ollama Cloud |
| `DB_PATH` | ❌ | `.checkpoints/checkpoints.db` | SQLite checkpoint database path |

> 🔒 **Security**: `.env` is in `.gitignore` — never commit secrets.

---

## TUI Usage (`app.py`)

Launch the interactive terminal interface:

```bash
uv run python app.py
```

You will see a rich banner and a numbered menu:

```
┌─────────────────────────────────────────────────────────────┐
│ GitHub Intelligence Agent                                   │
│ Multi-Server MCP + LangGraph + ChatOllama Cloud             │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│ Main Menu                                                   │
├─────────────────────────────────────────────────────────────┤
│ [1] Run Query                                               │
│ [2] Stream Query (live tokens)                              │
│ [3] Continue Thread                                         │
│ [4] View History                                            │
│ [5] Settings                                                │
│ [Q] Quit                                                    │
└─────────────────────────────────────────────────────────────┘
```

### 1. Run Query

Enter a natural-language instruction. The agent will:
1. Connect to the GitHub MCP server
2. Fetch available tools
3. Run the LangGraph ReAct loop (agent → tools → agent)
4. Print the final answer in a styled green panel

**Example queries:**
- `Summarize open PRs for langchain-ai/langchain`
- `Who are the top 5 contributors to facebook/react this month?`
- `List high-priority issues in vercel/next.js labeled 'bug'`

### 2. Stream Query (live tokens)

Same as **Run Query**, but displays results streamed in real-time as the model generates tokens. Uses `rich.live.Live` with a cyan border panel.

### 3. Continue Thread

Lists all persisted conversation threads from the SQLite checkpointer:

```
#  Thread ID              Messages 
--  -------------------  ----------
1   tui_default                  4
2   session_2024_05_12             8
```

Select a thread by number, then type a follow-up message. The agent resumes the conversation with full context.

### 4. View History

Shows a summary table of all checkpointed threads and their message counts, plus the path to the SQLite DB.

### 5. Settings

Displays a table of all environment variables and code defaults (sensitive values are masked for security):

| Key | Value | Source |
|---|---|---|
| GITHUB_PAT | `ghp_…xxxx` | env |
| OLLAMA_API_KEY | `sk_…xxxx` | env |
| OLLAMA_BASE_URL | `https://api.ollama.ai` | env |
| … | … | … |

### Q. Quit

Press **`Q`** at any menu to exit gracefully. `Ctrl+C` is also handled.

---

## CLI Usage (`agent.py`)

For quick one-shot queries without the TUI:

```bash
# Default query (analyzes langchain-ai/langchain)
uv run python agent.py

# Custom query
uv run python agent.py "List open issues in facebook/react and summarize them"
```

The result is printed as plain text to stdout.

---

## Architecture Overview

```text
┌────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│   User Input   │────▶│  app.py (TUI)    │────▶│   agent.py (core)   │
│  (Rich Prompt) │     │  - Menu / Prompt │     │  - Config / Graph   │
└────────────────┘     │  - Live Display  │     └──────────┬──────────┘
                       └──────────────────┘                │
                                                          │
                       ┌──────────────────────────────────▼────────────────┐
                       │              LangGraph StateGraph                  │
                       │  ┌─────────┐    ┌─────────┐    ┌─────────┐    │
                       │  │  START  │───▶│  agent  │───▶│ should_   │    │
                       │  └─────────┘    └─────────┘    │ continue? │    │
                       │                    ▲           └────┬────┘    │
                       │                    │                │         │
                       │              ┌────┴────┐      ┌──▼───┐     │
                       │              │  tools  │◀─────│ END  │     │
                       │              └─────────┘      └──────┘     │
                       └──────────────────────────────────────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    │                                           │
            ┌───────▼────────┐                          ┌──────▼──────┐
            │ MCP Client     │                          │ ChatOllama  │
            │ (GitHub HTTP)  │                          │ (Cloud API) │
            │ Bearer: GITHUB │                          │ host+key    │
            └────────────────┘                          └─────────────┘
                    │
            ┌───────▼────────┐
            │ Async SQLite   │
            │ Checkpointer   │
            │ (threads, mem) │
            └────────────────┘
```

### Component Breakdown

| Component | File | Description |
|-----------|------|-------------|
| **TUI Interface** | `app.py` | Rich-powered interactive menu, prompts, live displays |
| **Agent Core** | `agent.py` | Async LangGraph builder + `run_agent()` execution helper |
| **MCP Client** | `langchain_mcp_adapters` | `MultiServerMCPClient` wrapping the GitHub MCP HTTP server |
| **LLM** | `langchain_ollama` | `ChatOllama` connected to Ollama Cloud via `host` + `async_client_kwargs` |
| **Checkpointer** | `langgraph_checkpoint_sqlite` | `AsyncSqliteSaver` for persistent conversation threads |
| **Env Loader** | `python-dotenv` | Loads `.env` variables at startup |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GITHUB_PAT` | **Yes** | GitHub Personal Access Token with `repo` and `read:org` scopes |
| `OLLAMA_API_KEY` | **Yes** | API key from [Ollama Cloud](https://ollama.ai) |
| `OLLAMA_BASE_URL` | No | Ollama API base URL (default: `https://api.ollama.ai`) |
| `OLLAMA_MODEL` | No | Model identifier (default: `kimi-k2.6:cloud`) |
| `DB_PATH` | No | SQLite checkpoint database path (default: `.checkpoints/checkpoints.db`) |

### How `GITHUB_PAT` is used

The raw MCP config from the original `agent.py` snippet:

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

…is adapted in code to the `langchain-mcp-adapters` format:

```python
MultiServerMCPClient({
    "github": {
        "transport": "http",
        "url": "https://api.githubcopilot.com/mcp/",
        "headers": {"Authorization": f"Bearer {GITHUB_PAT}"},
    }
})
```

### How `OLLAMA_API_KEY` is used

`langchain-ollama` creates an `httpx.AsyncClient` under the hood. The `async_client_kwargs` are merged into that client, so custom headers (like `Authorization: Bearer …`) are forwarded on **every LLM request**:

```python
ChatOllama(
    model="kimi-k2.6:cloud",
    host="https://api.ollama.ai",
    async_client_kwargs={
        "headers": {"Authorization": f"Bearer {OLLAMA_API_KEY}"}
    },
)
```

---

## Troubleshooting

### `RuntimeError: Missing required environment variable: GITHUB_PAT`

Your `.env` file is missing `GITHUB_PAT`. Run:

```bash
echo 'GITHUB_PAT="ghp_xxxxxx"' >> .env
```

### `RuntimeError: Missing required environment variable: OLLAMA_API_KEY`

Same as above — add your Ollama Cloud API key to `.env`.

### MCP client fails to connect

1. Verify `GITHUB_PAT` has the correct scopes (`repo`, `read:org`).
2. Check network connectivity to `https://api.githubcopilot.com/mcp/`.
3. Ensure the token has not expired.

### LLM returns no content / errors

1. Verify `OLLAMA_API_KEY` is valid at [ollama.ai](https://ollama.ai).
2. Check `OLLAMA_BASE_URL` matches your cloud endpoint (default is `https://api.ollama.ai`).
3. Ensure the model name (`kimi-k2.6:cloud`) is available on your Ollama account.

### SQLite checkpointer errors

The `.checkpoints/` directory is created automatically. If you see permission errors:

```bash
mkdir -p .checkpoints
chmod 755 .checkpoints
```

---

## Extending the Agent

### Add more MCP servers

Extend `_build_mcp_config()` in `agent.py`:

```python
return {
    "github": {
        "transport": "http",
        "url": "https://api.githubcopilot.com/mcp/",
        "headers": {"Authorization": f"Bearer {GITHUB_PAT}"},
    },
    "filesystem": {
        "transport": "stdio",
        "command": "python",
        "args": ["-m", "mcp_server_filesystem", "/path/to/allowed"],
    },
}
```

### Swap the LLM

Change `OLLAMA_MODEL` in `.env`:

```env
OLLAMA_MODEL="llama3.1:cloud"
```

Or edit `_build_llm()` in `agent.py` for advanced tuning (temperature, `num_predict`, etc.).

### Add a web UI

Wrap `run_agent()` in a FastAPI endpoint or Streamlit app:

```python
# FastAPI example
from fastapi import FastAPI
from agent import run_agent

app = FastAPI()

@app.post("/query")
async def query_endpoint(q: str, thread_id: str = "default"):
    result = await run_agent(q, thread_id=thread_id)
    return {"result": result}
```

### Swap checkpointer backend

Replace `AsyncSqliteSaver` with `AsyncPostgresSaver` or `AsyncRedisSaver` for multi-instance deployments:

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

checkpointer = AsyncPostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost:5432/postgres?sslmode=disable"
)
```

---

## License

MIT

---

## Credits

Built with:
- 🦜️🔗 [LangGraph](https://github.com/langchain-ai/langgraph)
- 🔧 [langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters)
- 🦙 [langchain-ollama](https://github.com/langchain-ai/langchain-ollama)
- 🎨 [Rich](https://github.com/Textualize/rich)
