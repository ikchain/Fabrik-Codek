# Fabrik-Codek MCP Server - Design Document

**Date**: 2026-02-19
**Status**: Approved
**Phase**: 1 of 3 (MCP Server → OpenClaw Skill → DX/Onboarding)

## Context

Fabrik-Codek is a local AI dev assistant with hybrid RAG (vector + knowledge graph), a data flywheel, and a REST API. To reach broader adoption via OpenClaw marketplace and other agent ecosystems, we need to expose it as an MCP server.

## Goal

Expose Fabrik-Codek as a standard MCP server so any MCP-compatible agent (Claude Code, OpenClaw, Cursor, Windsurf, etc.) can use it as a local knowledge base. This becomes the technical foundation for the OpenClaw skill (Phase 2).

## Target Audience

- Individual developers wanting a private, local knowledge base for their agents
- Small teams (2-10) wanting shared knowledge (future premium feature)

## Architecture

### MCP Server wrapping existing Fabrik services

```
Agent (Claude Code / OpenClaw / Cursor)
    │
    ▼
MCP Protocol (stdio / SSE)
    │
    ▼
Fabrik MCP Server (src/interfaces/mcp_server.py)
    │
    ├── Tools (actions agents can invoke)
    │   ├── fabrik_ask        → LLM + optional RAG context
    │   ├── fabrik_search     → Vector semantic search
    │   ├── fabrik_graph_search → Knowledge graph traversal
    │   ├── fabrik_graph_stats  → Graph statistics
    │   └── fabrik_status     → System health check
    │
    ├── Resources (data agents can read)
    │   ├── fabrik://status          → System status
    │   ├── fabrik://graph/stats     → Graph statistics
    │   └── fabrik://config          → Current configuration
    │
    └── Existing Services (unchanged)
        ├── LLMClient (Ollama)
        ├── HybridRAGEngine (LanceDB + NetworkX)
        ├── GraphEngine
        └── FlywheelCollector
```

### Transport

- **stdio** (default): For local use with Claude Code, Cursor, etc.
- **SSE**: For network access (team use, OpenClaw remote)

### Tools Exposed

| Tool | Description | Maps to |
|------|-------------|---------|
| `fabrik_ask` | Ask with optional RAG/graph context | POST /ask |
| `fabrik_search` | Semantic search in knowledge base | POST /search |
| `fabrik_graph_search` | Search knowledge graph entities | POST /graph/search |
| `fabrik_graph_stats` | Get graph statistics | GET /graph/stats |
| `fabrik_status` | Check system health | GET /status |

### Resources Exposed

| URI | Description |
|-----|-------------|
| `fabrik://status` | System component status |
| `fabrik://graph/stats` | Knowledge graph statistics |
| `fabrik://config` | Current Fabrik configuration (sanitized) |

## Key Decisions

1. **MCP SDK**: Use official `mcp` Python SDK (maintained by Anthropic)
2. **No new dependencies beyond `mcp`**: Reuse existing services
3. **Separate entry point**: `fabrik mcp` CLI command (not replace existing `fabrik serve`)
4. **REST API stays**: MCP server is additive, not a replacement
5. **Flywheel capture from MCP**: MCP interactions also feed the data flywheel

## Monetization Path

- **Free**: Individual use, all tools, stdio transport
- **Premium** (future): Team mode with SSE, shared knowledge base, usage analytics

## Success Criteria

1. `fabrik mcp` starts an MCP server on stdio
2. Claude Code can use it via settings.json MCP config
3. All 5 tools work correctly
4. Existing tests still pass (413)
5. New tests for MCP server (tool calls, resources)
