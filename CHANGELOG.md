# Changelog

All notable changes to Fabrik-Codek are documented in this file.

## [Unreleased]

_Nothing yet._

## [1.3.0] - 2026-08-16

### Added
- **Related Work section in README** — Reference to CASK benchmark (Kaggle × Google DeepMind *Measuring AGI* hackathon, 2026). Evaluation of 17 frontier LLMs on context sensitivity and metacognitive calibration. Links to writeup and notebook on Kaggle.
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

- **Context Gate** — Heuristic gate deciding whether to inject retrieval context
  - 4 signals: entity density, competence level, task type rules, topic presence
  - Always-inject override for debugging/code_review/architecture
  - Prevents RAG contamination for simple or off-domain queries
- **Context Map** — Deterministic routing bypass for predictable queries
  - Regex-based pattern matching skips full Router + RAG pipeline
  - Configurable per entry: task_type, use_context, profile_fragments
  - Inspired by [Savia](https://github.com/gonzalezpazmonica/pm-workspace)'s context engineering
- **Profile Fragmentation** — Selective profile loading per task type
  - `PersonalProfile.to_fragment(name)`: identity, tech_stack, patterns, projects, style, decisions
  - `TASK_PROFILE_FRAGMENTS` maps each task type to relevant fragments
  - Reduces system prompt noise by loading only what's needed
- **U-Shape Prompt Positioning** — System prompt reordered for optimal LLM attention
  - Order: task instruction → personal profile → competence fragment
  - Based on Liu et al. 2024 "Lost in the Middle": critical info at start and end
- **Context Aging** — Temporal decay on RAG chunk scores
  - `_compute_age_factor()`: exponential half-life formula with configurable min_factor
  - Applied between RRF fusion and relevance filter
  - Config: `FABRIK_AGING_HALF_LIFE_DAYS` (default: 90)
- **Instincts Protocol** — Emergent behavioral patterns with variable confidence
  - `Instinct` dataclass: reinforce (+0.03), penalize (-0.05), decay (-0.05/30 days)
  - `InstinctRegistry`: match, reinforce, penalize, apply_decay_all, stats
  - 5 categories: workflow, preference, shortcut, context, timing
  - Inspired by [Savia](https://github.com/gonzalezpazmonica/pm-workspace)'s synaptic learning
- **Agent Mode MCP** — Machine-to-machine structured output
  - `fabrik_ask_agent` tool: JSON envelope with status (OK/ERROR/NO_CONTEXT)
  - Separated answer, metadata (model, tokens, latency), and sources
- **Per-Task Token Control** — `max_tokens` per task type in RetrievalStrategy
  - debugging/architecture/ml_engineering: 1536
  - code_review/explanation/testing/devops: 1024
  - general: 768
- **Post-Retrieval Relevance Filter** — Token-overlap filter drops irrelevant RAG chunks
  - Jaccard-like on meaningful tokens (bilingual stopwords removed)
  - Applied after RRF fusion, before graph context enrichment

### Changed
- All CLI messages, prompts, and logger errors translated from Spanish to English
- `HybridRAGEngine._rrf_fusion()` extended to accept optional fulltext results
- Multi-source origin tracking: results found in multiple sources tagged as `"hybrid"`
- **Version numbering unified.** The package, the module and this file had drifted apart —
  `pyproject.toml` said `0.1.0`, `src/__init__.py` (the version the API reports on `/health`
  and `/status`) said `0.2.0`, and this changelog was still on `1.2.1`. All three now track
  the same number.

### Security
- **SQL injection guard for LanceDB filters** — `src/knowledge/sql_utils.py` escapes and
  validates every value interpolated into a `.where()` clause. String interpolation into
  those filters was reachable from user-controlled input in both the vector store and the
  retrieval layer.

### Fixed
- **Deterministic benchmark runs** — `run_eval_benchmark.py` called the `ollama run` CLI,
  which accepts no sampling parameters and therefore inherited the Modelfile temperature
  (typically 0.7–0.8). Benchmark runs were consequently irreproducible: the same model over
  the same cases scored differently each time, by a margin that can exceed the difference
  between two models being compared. Now uses the HTTP API with `temperature=0`, `top_k=1`
  and a fixed seed. Sanity check documented in the module: run the same model twice; the
  scores must be identical.
- **`temperature=0` was silently ignored** — `LLMClient` used `temperature or default`,
  and `0` is falsy, so requests asking for deterministic output got the default instead.
- **Naive/aware datetime mixing** — `src/utils/time.py` (`now_utc`, `ensure_aware`) applied
  at every subtraction site in RAG aging, graph decay and instinct decay. Previously these
  could raise `TypeError` when stored timestamps predated the convention.
- **Non-atomic writes** — `src/utils/io.py` (`atomic_write_json` / `atomic_write_text`,
  tempfile + `os.replace` + fsync) for full-file writers, plus a buffer-loss fix in the
  feedback collector's flush path. An interrupted write could previously truncate state.
- **Truncated hash IDs** — `src/utils/hashing.py` moves record IDs from truncated MD5 to
  SHA256, widening the space enough to make collisions a non-issue.
- **Ollama health check went stale** — an early return bypassed the TTL, so a backend that
  recovered (or died) after the first probe kept reporting its old state indefinitely.
- **RAG engine race under concurrency** — `get_rag_engine` is now task-safe via a lazy
  `asyncio.Lock` with publish-after-init, so concurrent `asyncio.gather` callers cannot
  observe a half-initialised engine.
- **MAB reward attribution** — the bandit only records a reward when the strategy actually
  executed was the one it selected; an evolved strategy or an override overriding it now
  clears the arm id instead of corrupting the arm's statistics.
- **Profile and competence caches never invalidated** — both now key off file mtime.

### Documentation
- **README leads with the Personalization Paradox** — the ablation that showed the full
  personalization pipeline scoring *below* the raw model, the three causes it isolated,
  the components built in response, and the post-mitigation result, with limitations
  stated (single runs, N=50, no confidence intervals). Links the published study.
- **Reliability & Security section** in the README covering the hardening above.
- Corrected the routing description: it claimed competence-based escalation to a larger
  fallback model, which has been disabled since the ablation showed that layer subtracting.

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
