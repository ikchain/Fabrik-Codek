# Changelog

All notable changes to Fabrik-Codek are documented in this file.

## [Unreleased]

### Added
- **Thompson Sampling Strategy Optimizer** — Multi-Armed Bandit for retrieval strategy selection
  - 4 discrete arms: default, graph_boost, deep_graph, vector_focus
  - Beta distributions updated with outcome feedback (accepted/rejected)
  - Graceful fallback to static overrides when < 5 samples per arm
  - Integrated into CLI chat/ask and `fabrik competence build`
- **Adaptive Forgetting Curve** — Spaced repetition for knowledge decay
  - `effective_half_life = base * (1 + 0.3 * reinforcement_count)^0.5`
  - Reinforced entities decay slower; neglected entities fade faster
  - Integrated into `apply_decay()` with backward-compatible edge format
- **Graph-Guided Chunk Expansion (KG2RAG)** — Better graph retrieval via composite queries
  - Builds expansion queries from graph neighborhoods (e.g., "FastAPI Pydantic")
  - Deduplicates symmetric pairs, sorts by proximity score
  - Replaces entity-name-only graph search with context-rich composite queries
- **Incremental Profile Build (SPRInG)** — Update profile without full datalake rescan
  - Timestamp filtering: only analyzes entries since last build
  - Drift detection: cosine distance between old and new topic distributions
  - EMA merge: alpha=0.7 on drift, 0.3 on gradual change
  - Replay buffer: stores top-20 most novel new entries
  - CLI: `fabrik profile build-incremental` and `fabrik profile drift`
- **Adaptive Retrieval (Stop-RAG)** — Confidence-based stopping for vector search
  - `retrieve_adaptive()`: fetches max_k results but returns only what's needed
  - Combined confidence: `0.6 * similarity_avg + 0.4 * entity_coverage`
  - Per-task thresholds in RetrievalStrategy (e.g., debugging=0.6, explanation=0.8)
  - Integrated into HybridRAGEngine when `confidence_threshold` is set
- **Learned Task Router (TF-IDF)** — Classification learned from accepted outcomes
  - TF-IDF vectorizer with centroid-per-task-type classification
  - 3-level chain: learned (confidence >= 0.3) → keywords → LLM fallback
  - Corpus auto-built from outcomes during `fabrik competence build`
  - sklearn import-guarded: degrades gracefully when unavailable
  - Loaded in all entry points: CLI chat/ask, API, MCP, and router test
- **Graph Temporal Decay** — Exponential weight decay for knowledge graph edges
  - Edges store `base_weight` and `last_reinforced` timestamps; entities store `last_seen`
  - `apply_decay()`: idempotent formula `weight = base_weight * 0.5^(days/half_life)`
  - Integrated into pipeline build (runs after graph completion, before save)
  - CLI: `fabrik graph decay --dry-run --half-life 90`
  - Config: `FABRIK_GRAPH_DECAY_HALF_LIFE_DAYS` (default: 90)
- **Competence Model** — Knowledge depth scoring per topic
  - `CompetenceBuilder` analyzes 4 signals: entry count (log scale), entity density (graph), recency (exponential decay), outcome rate
  - 8 adaptive weight sets with graceful degradation when signals are missing
  - Competence levels: Expert (>=0.8), Competent (>=0.4), Novice (>=0.1), Unknown (<0.1)
  - CLI: `fabrik competence build` and `fabrik competence show`
- **Meilisearch full-text search** — Optional BM25-style keyword search as third retrieval tier
  - `FullTextEngine` async wrapper using httpx (no new dependencies)
  - Three-tier Reciprocal Rank Fusion (RRF): vector + graph + fulltext
  - CLI: `fabrik fulltext status|index|search`
  - Graceful degradation — works without Meilisearch (`fulltext_weight=0.0` by default)
- **Adaptive Task Router** — Intelligent query classification and routing
  - 3-level classification chain: learned TF-IDF → keyword matching → LLM fallback
  - Topic detection from CompetenceMap with automatic model escalation
  - Per-task retrieval strategies with confidence thresholds
  - 3-layer system prompt: personal profile + competence + task-specific instructions
  - CLI: `fabrik router test -q "query"` for classification debugging
  - Integrated into CLI (`ask`, `chat`), API (`/ask`), and MCP (`fabrik_ask`)

### Changed
- All CLI messages, prompts, and logger errors translated from Spanish to English
- `HybridRAGEngine._rrf_fusion()` extended to accept optional fulltext results
- Multi-source origin tracking: results found in multiple sources tagged as `"hybrid"`
- Test count: 527 → 957

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
