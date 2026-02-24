---
name: fabrik-codek
description: "Cognitive architecture for developers: hyper-personalization engine (personal profiling, competence model, adaptive task routing, outcome tracking) over three-tier hybrid retrieval (vector + knowledge graph + full-text). A 7B model that knows you is worth more than a 400B that doesn't. 100% local, zero cloud dependencies."
version: 1.6.0
homepage: https://github.com/ikchain/Fabrik-Codek
user-invocable: true
metadata:
  clawdbot:
    requires:
      bins: [fabrik]
      anyBins: [python3, python]
    homepage: https://github.com/ikchain/Fabrik-Codek
    os: [macos, linux]
    emoji: "🧠"
    configPaths:
      - "~/.claude/projects/*"
      - "./data/embeddings/"
      - "./data/graphdb/"
    install:
      - kind: pip
        package: fabrik-codek
        bins: [fabrik]
---

# Fabrik-Codek - Local Cognitive Architecture for Developers

Fabrik-Codek is a **cognitive architecture** — a system where perception, memory, reasoning, learning, and action work together, much like how a human developer accumulates expertise over time. Unlike plain RAG tools that just retrieve text, Fabrik-Codek combines three retrieval tiers — vector search (semantic), knowledge graph traversal (relational), and full-text search (keyword/BM25) — fused via Reciprocal Rank Fusion (RRF). It continuously improves through a data flywheel that captures what you do and feeds it back into every future query.

**How it works**: When you run `fabrik learn process`, Fabrik-Codek reads your local Claude Code session transcript files (`~/.claude/projects/*/` — JSON files already on your disk) and extracts structured knowledge (patterns, decisions, debugging strategies). It stores this in a local vector DB (LanceDB, in `./data/embeddings/`) and a local knowledge graph (NetworkX, in `./data/graphdb/`). When you query via MCP tools, it uses hybrid retrieval to give your AI agent deep project context — not just keyword matches, but an understanding of how concepts in your codebase connect. No data leaves your machine at any point.

## Setup

Fabrik-Codek runs as an MCP server. Configure it in your `openclaw.json`:

```json
{
  "mcpServers": {
    "fabrik-codek": {
      "command": "fabrik",
      "args": ["mcp"]
    }
  }
}
```

Or for network access (SSE transport):

```json
{
  "mcpServers": {
    "fabrik-codek": {
      "command": "fabrik",
      "args": ["mcp", "--transport", "sse", "--port", "8421"]
    }
  }
}
```

## Available Tools

### fabrik_search
Semantic vector search in the knowledge base. Use this when you need to find relevant documents, patterns, or examples from accumulated project knowledge.

Example: "Search my knowledge base for repository pattern implementations"

### fabrik_graph_search
Search the knowledge graph for entities (technologies, patterns, strategies) and their relationships. Use this to understand how concepts connect.

Example: "Find entities related to FastAPI in the knowledge graph"

### fabrik_fulltext_search
Full-text keyword search via Meilisearch. Use this for exact keyword or phrase matching when you know the specific terms you're looking for. Requires Meilisearch running locally (optional — system works without it).

Example: "Search for 'retry exponential backoff' in the knowledge base"

### fabrik_ask
Ask a coding question to the local LLM with optional context from the knowledge base. Set `use_rag=true` for vector search context or `use_graph=true` for hybrid (vector + graph + fulltext) context.

Example: "Ask fabrik how to implement dependency injection using knowledge base context"

### fabrik_graph_stats
Get statistics about the knowledge graph: entity counts, relationship types, and graph density.

### fabrik_status
Check system health: Ollama availability, RAG engine, knowledge graph, full-text search, and datalake status.

### fabrik_profile
Build or view your personal profile. The profile analyzes your datalake and generates behavioral system prompt instructions so the LLM responds using your actual stack and preferences.

Example: "Build my profile" or "Show my profile"

## When to Use

- **Need project context?** Use `fabrik_search` for semantic similarity or `fabrik_fulltext_search` for exact keyword matching
- **Exploring relationships?** Use `fabrik_graph_search` to traverse the knowledge graph
- **Coding question?** Use `fabrik_ask` with `use_rag` or `use_graph` for context-enriched answers
- **Checking setup?** Use `fabrik_status` to verify all components are running

## Requirements

- [Fabrik-Codek](https://github.com/ikchain/Fabrik-Codek) installed (`pip install fabrik-codek`)
- [Ollama](https://ollama.ai/) running locally with a model pulled (e.g., `ollama pull qwen2.5-coder:7b`)

## Security & Privacy

- **100% local**: All data stays on your machine. No external API calls, no telemetry, no cloud dependencies.
- **No credentials required**: Fabrik-Codek connects only to your local Ollama instance (`localhost:11434`).
- **External endpoints**: None. This skill does not contact any external services.
- **Data paths**: Reads transcript files from `~/.claude/projects/*/` (local JSON already on disk). Writes indexed data to `./data/embeddings/` (vector DB) and `./data/graphdb/` (knowledge graph). Both paths are declared in the skill metadata.
- **Session reading**: The `fabrik learn` command is opt-in — triggered manually by the user, not automatic background surveillance. Transcripts may contain sensitive session data; review before indexing.
- **Network exposure**: Default transport is `stdio` (no network). SSE transport (`--transport sse`) binds to `127.0.0.1` by default. If you change the bind address, ensure proper firewall/ACL rules to avoid exposing indexed data over the network.
- **Install source**: Fully open source at [github.com/ikchain/Fabrik-Codek](https://github.com/ikchain/Fabrik-Codek) (MIT license). Verify the pip package source matches the GitHub repository before installing.
