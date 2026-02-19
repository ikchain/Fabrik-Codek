---
name: fabrik-codek
description: Local AI knowledge base powered by hybrid RAG (vector + knowledge graph). Connects any Ollama model to your accumulated project knowledge. Privacy-first, runs 100% locally.
homepage: https://github.com/ikchain/Fabrik-Codek
user-invocable: true
metadata: {"openclaw": {"requires": {"bins": ["fabrik"]}, "emoji": "🧠"}}
---

# Fabrik-Codek - Local AI Knowledge Base

Fabrik-Codek is a local AI dev assistant that combines vector search (LanceDB) with a knowledge graph (NetworkX) to provide context-aware coding assistance. It learns from your interactions via a data flywheel and feeds that knowledge back into every query.

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

### fabrik_ask
Ask a coding question to the local LLM with optional context from the knowledge base. Set `use_rag=true` for vector search context or `use_graph=true` for hybrid (vector + graph) context.

Example: "Ask fabrik how to implement dependency injection using knowledge base context"

### fabrik_graph_stats
Get statistics about the knowledge graph: entity counts, relationship types, and graph density.

### fabrik_status
Check system health: Ollama availability, RAG engine, knowledge graph, and datalake status.

## When to Use

- **Need project context?** Use `fabrik_search` to find relevant knowledge
- **Exploring relationships?** Use `fabrik_graph_search` to traverse the knowledge graph
- **Coding question?** Use `fabrik_ask` with `use_rag` or `use_graph` for context-enriched answers
- **Checking setup?** Use `fabrik_status` to verify all components are running

## Requirements

- [Fabrik-Codek](https://github.com/ikchain/Fabrik-Codek) installed (`pip install fabrik-codek`)
- [Ollama](https://ollama.ai/) running locally with a model pulled (e.g., `ollama pull qwen2.5-coder:7b`)
