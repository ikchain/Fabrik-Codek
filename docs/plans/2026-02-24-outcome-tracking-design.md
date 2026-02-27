# Outcome Tracking

**Date:** 2026-02-24
**Status:** Draft
**Depends on:**  (Adaptive Task Router),  (Competence Model)

## Problem

Fabrik-Codek has a complete adaptive pipeline (Personal Profile, Competence
Model, Task Router, Graph Temporal Decay) but no way to know if its responses
are actually useful. The cognitive loop is open: the system adapts HOW it
responds but never learns WHETHER its responses worked.

Without outcome data:
- Competence scores are based on data volume, not response quality
- Retrieval strategies are static defaults per task type, never refined
- There is no objective metric for "is Fabrik getting better?"

## Design Principles

1. **Zero friction** — No manual feedback, no prompts, no UX changes
2. **Observe, don't ask** — Infer outcomes from conversational patterns
3. **Conservative inference** — Prefer false-accepted over false-rejected
4. **Statistical emergence** — Individual outcomes may be noisy; aggregate
   patterns are reliable with sufficient volume
5. **Graceful degradation** — System works identically with zero outcomes
   (falls back to existing 3-signal competence model)

## Scope

**In scope (this ticket):**
- OutcomeTracker with conversational pattern inference (CLI chat + ask)
- Outcome persistence to datalake
- Competence Model integration (4th signal: outcome_rate)
- StrategyOptimizer (adjusts retrieval per task_type + topic)
- TaskRouter integration (loads strategy overrides)
- CLI `fabrik outcomes` command
- Tests for all new and modified components

**Out of scope:**
- API endpoint integration (future)
- MCP server integration (future)
- LLM-based outcome inference (overkill for now)

## Data Model

### OutcomeRecord

```python
@dataclass
class OutcomeRecord:
    id: str                    # uuid4
    timestamp: str             # ISO 8601
    query: str                 # user's original question
    response_summary: str      # first 200 chars of LLM response
    task_type: str             # from RoutingDecision
    topic: str | None          # from RoutingDecision
    competence_level: str      # from RoutingDecision
    model: str                 # model used for generation
    strategy: dict             # RetrievalStrategy as dict
    outcome: str               # "accepted" | "rejected" | "neutral"
    inference_reason: str      # why this outcome was inferred
    latency_ms: float          # LLM response latency
    session_id: str            # groups outcomes by session
```

### Storage

File: `datalake/01-raw/outcomes/YYYY-MM-DD_outcomes.jsonl`

One JSON object per line, append-only. Same pattern as auto-captures.

## Outcome Inference

### Conversational Pattern Matching

The tracker holds a `pending_turn` with the previous turn's data. When the
next user message arrives, it infers the outcome of the pending turn:

| Pattern | Outcome | Detection Method |
|---------|---------|-----------------|
| User changes topic | `accepted` | Low token overlap (< 0.5) with previous query |
| Constructive follow-up | `accepted` | Low overlap + no negation keywords |
| User rephrases same question | `rejected` | High token overlap (>= 0.5) with previous query |
| Explicit negation | `rejected` | Negation keywords: "no", "incorrecto", "mal", "eso no es", "wrong", "incorrect" |
| Session end (exit/quit/ctrl+d) | `neutral` | Session close event |
| Single-shot (fabrik ask) | `neutral` | No follow-up possible, reason: "single_shot" |

### Similarity Function

Token overlap between previous and new query:

```python
def token_similarity(query_a: str, query_b: str) -> float:
    tokens_a = set(query_a.lower().split())
    tokens_b = set(query_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    smaller = min(len(tokens_a), len(tokens_b))
    return len(intersection) / smaller
```

Threshold: 0.5. Conservative — prefers false `accepted` over false `rejected`
because a wrong `rejected` contaminates more than a wrong `accepted`.

### Negation Keywords

Two lists (Spanish + English) since user works in both:

```python
NEGATION_KEYWORDS = [
    "no", "incorrecto", "mal", "eso no es", "eso no",
    "wrong", "incorrect", "bad", "that's not", "nope",
]
```

Checked against the start of the new message (first 50 chars, lowercased).
Only triggers if the token similarity is also moderate (> 0.3) — prevents
false rejection when user says "no" about an unrelated topic.

## Components

### 1. OutcomeTracker

**File:** `src/flywheel/outcome_tracker.py`

Synchronous. Stateful (holds pending_turn). One instance per session.

```python
class OutcomeTracker:
    def __init__(self, datalake_path: Path, session_id: str): ...

    def record_turn(
        self,
        query: str,
        response: str,
        decision: RoutingDecision,
        latency_ms: float,
    ) -> OutcomeRecord | None:
        """Record a turn. Returns OutcomeRecord for the PREVIOUS turn if one
        was pending, or None if this is the first turn."""

    def close_session(self) -> OutcomeRecord | None:
        """Close session. Marks pending turn as neutral and returns it."""

    def get_session_stats(self) -> dict:
        """Return {total, accepted, rejected, neutral} for this session."""
```

Internal flow of `record_turn`:
1. If `pending_turn` exists, infer outcome by comparing `pending_turn.query`
   with the new `query`
2. Persist the outcome (append to JSONL)
3. Store current turn as new `pending_turn`
4. Return the OutcomeRecord of the previous turn (or None)

### 2. StrategyOptimizer

**File:** `src/core/strategy_optimizer.py`

Stateless. Reads outcomes from datalake, produces strategy overrides.

```python
class StrategyOptimizer:
    def __init__(self, datalake_path: Path): ...

    def compute_overrides(self, days: int = 30) -> dict[str, RetrievalStrategy]:
        """Read outcomes from last N days, compute acceptance rates per
        (task_type, topic), return strategy overrides for underperforming
        combinations."""

    def save_overrides(self, output_path: Path) -> int:
        """Compute and save overrides. Returns number of overrides generated."""
```

**Override logic per (task_type, topic) with >= 10 outcomes:**

| Acceptance Rate | Action |
|----------------|--------|
| >= 0.7 | Keep current strategy (working well) |
| 0.5 - 0.7 | `graph_depth` +1, `graph_weight` +0.1 |
| < 0.5 | `graph_depth` +2, `graph_weight` +0.2, `fulltext_weight` = 0.1 |

**Storage:** `data/profile/strategy_overrides.json`

```json
{
    "explanation_kubernetes": {
        "graph_depth": 4,
        "graph_weight": 0.6,
        "fulltext_weight": 0.1,
        "acceptance_rate": 0.33,
        "sample_size": 12,
        "updated_at": "2026-02-24T10:00:00"
    }
}
```

### 3. CompetenceModel Changes

**File:** `src/core/competence_model.py` (modified)

New 4th signal: `outcome_rate` — acceptance rate per topic from outcomes.

**Weight sets updated:**

| Weight Set | entry_count | entity_density | recency | outcome_rate |
|-----------|-------------|----------------|---------|-------------|
| ALL | 0.30 | 0.25 | 0.20 | 0.25 |
| NO_GRAPH | 0.40 | — | 0.30 | 0.30 |
| NO_RECENCY | 0.40 | 0.30 | — | 0.30 |
| ENTRY_ONLY | 0.60 | — | — | 0.40 |

**Graceful degradation:** If a topic has < 5 outcomes, the outcome_rate signal
is omitted and weights redistribute among existing signals — same pattern
already used for graph and recency signals.

**Minimum threshold:** 5 outcomes per topic to activate the signal.

### 4. TaskRouter Changes

**File:** `src/core/task_router.py` (modified)

At construction, load `data/profile/strategy_overrides.json` if it exists.

When building a `RoutingDecision`, check if an override exists for the
detected `(task_type, topic)`. If so, use the override strategy instead of
the default.

Fallback chain: override → task-type default → global default.

### 5. CLI Integration

**File:** `src/interfaces/cli.py` (modified)

#### `chat` command

```python
# At chat loop start:
tracker = OutcomeTracker(settings.datalake_path, session_id)

# After each response:
outcome = tracker.record_turn(
    query=user_input,
    response=response.content,
    decision=decision,
    latency_ms=response.latency_ms,
)

# At session end:
tracker.close_session()
```

#### `ask` command

```python
tracker = OutcomeTracker(settings.datalake_path, session_id)
tracker.record_turn(query=prompt, response=response.content,
                    decision=decision, latency_ms=response.latency_ms)
tracker.close_session()  # single-shot → neutral
```

#### New command: `fabrik outcomes`

```
fabrik outcomes show                        # Last 20 outcomes
fabrik outcomes stats                       # Aggregated stats
fabrik outcomes stats --topic postgresql    # Filtered by topic
fabrik outcomes stats --task-type debugging # Filtered by task type
```

#### Extended: `fabrik competence build`

After rebuilding competence scores, also run StrategyOptimizer to generate
strategy overrides. Single command rebuilds everything.

## Files Changed

| File | Change |
|------|--------|
| `src/flywheel/outcome_tracker.py` | **NEW** — OutcomeTracker, OutcomeRecord, infer_outcome |
| `src/core/strategy_optimizer.py` | **NEW** — StrategyOptimizer |
| `src/core/competence_model.py` | **MODIFIED** — add outcome_rate as 4th signal |
| `src/core/task_router.py` | **MODIFIED** — load strategy_overrides |
| `src/interfaces/cli.py` | **MODIFIED** — integrate tracker in chat/ask + outcomes command |
| `tests/test_outcome_tracker.py` | **NEW** |
| `tests/test_strategy_optimizer.py` | **NEW** |
| `tests/test_competence_model.py` | **MODIFIED** — tests for 4th signal |
| `tests/test_task_router.py` | **MODIFIED** — tests for strategy overrides |

## Success Criteria

- `fabrik chat` captures outcomes silently without UX changes
- `fabrik outcomes stats` shows acceptance rates per task_type + topic
- `fabrik competence build` integrates outcome_rate as 4th signal
- Strategy overrides are generated for underperforming combinations
- TaskRouter uses overrides when available
- All existing tests still pass (712+)
- New tests cover: inference logic, tracker lifecycle, optimizer thresholds,
  competence integration, router override loading
