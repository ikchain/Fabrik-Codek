# Changelog

All notable changes to Fabrik-Codek are documented in this file.

## [Unreleased]

### Added
- **Graph Temporal Decay** — Exponential weight decay for knowledge graph edges (FC-39)
  - Edges store `base_weight` and `last_reinforced` timestamps; entities store `last_seen`
  - `apply_decay()`: idempotent formula `weight = base_weight * 0.5^(days/half_life)`
  - Integrated into pipeline build (runs after graph completion, before save)
  - CLI: `fabrik graph decay --dry-run --half-life 90`
  - Config: `FABRIK_GRAPH_DECAY_HALF_LIFE_DAYS` (default: 90)
  - 11 new tests (4 timestamps + 7 decay including idempotency and prune integration)
- **Competence Model** — Knowledge depth scoring per topic (FC-36)
  - `CompetenceBuilder` analyzes 3 signals: entry count (log scale), entity density (graph), recency (exponential decay)
  - 4 adaptive weight sets with graceful degradation when signals are missing
  - Competence levels: Expert (>=0.8), Competent (>=0.4), Novice (>=0.1), Unknown (<0.1)
  - System prompt injection: competence fragment appended after Personal Profile
  - CLI: `fabrik competence build` and `fabrik competence show`
  - Persistence with JSON caching (`data/profile/competence_map.json`)
  - 89 new tests
- **Meilisearch full-text search** — Optional BM25-style keyword search as third retrieval tier
  - `FullTextEngine` async wrapper using httpx (no new dependencies)
  - Three-tier Reciprocal Rank Fusion (RRF): vector + graph + fulltext
  - CLI: `fabrik fulltext status|index|search`
  - API: `POST /fulltext/search` endpoint
  - MCP: `fabrik_fulltext_search` tool
  - 4 new config settings: `FABRIK_MEILISEARCH_URL`, `FABRIK_MEILISEARCH_KEY`, `FABRIK_MEILISEARCH_INDEX`, `FABRIK_FULLTEXT_WEIGHT`
  - Graceful degradation — works without Meilisearch (`fulltext_weight=0.0` by default)
  - 55 new tests (29 fulltext engine + 8 hybrid RRF + 8 MCP + 4 API + 6 CLI)
- **Adaptive Task Router** — Intelligent query classification and routing (FC-37)
  - Hybrid classification: keyword matching (7 task types) + LLM fallback
  - Topic detection from CompetenceMap with automatic model escalation
  - Per-task retrieval strategies (graph depth, vector/graph weights)
  - 3-layer system prompt: personal profile + competence + task-specific instructions
  - Model escalation: Novice/Unknown topics automatically use fallback model
  - CLI: `fabrik router test -q "query"` for classification debugging
  - Integrated into CLI (`ask`, `chat`), API (`/ask`), and MCP (`fabrik_ask`)
  - 63 new tests (data model + classification + topic + strategy + escalation + prompt + LLM + integration)

### Changed
- All CLI messages, prompts, and logger errors translated from Spanish to English
- `HybridRAGEngine._rrf_fusion()` extended to accept optional fulltext results
- Multi-source origin tracking: results found in multiple sources tagged as `"hybrid"`
- Test count: 527 → 711

## [1.2.1] - 2026-02-19

### Added
- Graph pruning: `fabrik graph prune` with `--dry-run`, `--min-mentions`, `--min-weight`, `--keep-inferred`
- 17 new tests for pruning (13 engine + 4 CLI)

## [1.2.0] - 2026-02-19

### Added
- Cognitive Architecture section in README with Mermaid diagram
- ClawHub/OpenClaw skill packaging (`skills/fabrik-codek/SKILL.md`)
- `configPaths` metadata for OpenClaw security scan compliance

## [1.1.0] - 2026-02-18

### Added
- MCP server (`fabrik mcp`) with stdio and SSE transport
- REST API (`fabrik serve`) with 7 endpoints
- `fabrik init` onboarding command
- API key authentication (optional)

## [1.0.0] - 2026-02-17

### Added
- Initial open-source release
- Hybrid RAG engine (vector + knowledge graph)
- Knowledge Graph with extraction pipeline (6 steps)
- Graph completion (transitive inference)
- Data flywheel with quality-gated logger
- Session observer for Claude Code transcripts
- CLI with 10+ commands
- 455 tests
