# Fix mark_feedback Persistence

**Date:** 2026-02-26
**Status:** Approved
**Depends on:**  (Outcome Tracking)

## Problem

`FlywheelCollector.mark_feedback()` only updates records in `self._buffer`
(in-memory). Once the buffer is flushed to disk, feedback is silently lost.
The method has a `# TODO: Update in persisted storage` comment at line 166.

Additionally, `mark_feedback` has **zero callers** in the entire codebase (CLI,
API, MCP). The method exists but is dead code. `export_training_pairs()` filters
by `user_feedback` but that field is always `"none"` because no feedback ever
reaches persisted records.

Meanwhile, `OutcomeTracker` already infers accepted/rejected outcomes
from conversational patterns but writes to a separate `outcomes/` directory
with no connection to `InteractionRecord.user_feedback`.

## Design Principles

1. **Append-only** — Never rewrite existing JSONL files (lesson from 2026-02-06
   data loss incident)
2. **Zero technical debt** — If the method exists, it must work AND be used
3. **Bridge existing systems** — Connect OutcomeTracker inference to
   InteractionRecord feedback via mark_feedback
4. **Graceful degradation** — Sidecar write failure must not crash the main flow

## Solution: Feedback Sidecar File

### Approach

Instead of rewriting interaction JSONL files in-place, feedback is persisted
to a separate sidecar directory (`feedback/`). When `export_training_pairs()`
runs, it merges both sources (interactions + feedback) by `record_id`.

### Data Model

New dataclass in `collector.py`:

```python
@dataclass
class FeedbackRecord:
    record_id: str          # ID of the original InteractionRecord
    feedback: str           # "positive" | "negative" | "neutral"
    was_edited: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "manual"  # "manual" | "outcome_tracker"
```

### Directory Structure

```
{datalake}/01-raw/
├── interactions/          # Existing — untouched
│   └── interactions_2026-02-26_abc12345.jsonl
├── feedback/              # NEW — sidecar feedback records
│   └── 2026-02-26_feedback.jsonl
└── outcomes/              # Existing — OutcomeTracker output
    └── 2026-02-26_outcomes.jsonl
```

### Persistence Flow

```
mark_feedback(record_id, feedback, source="manual")
    │
    ├─ 1. Search self._buffer (in-memory)
    │     → If found: update in place, return (existing behavior)
    │
    └─ 2. If not in buffer (already flushed):
          └─ Append FeedbackRecord to feedback/{YYYY-MM-DD}_feedback.jsonl
```

### Read Flow (export_training_pairs)

```
1. Load feedback index: read all feedback/*.jsonl
   → dict[record_id → FeedbackRecord] (last-write-wins by timestamp)

2. For each InteractionRecord in interactions/*.jsonl:
   → If record.id in feedback_index:
       override user_feedback and was_edited from FeedbackRecord
   → Apply existing min_feedback filter
```

### OutcomeTracker Bridge

In `cli.py` chat loop, after `tracker.record_turn()`:

```
OutcomeTracker infers outcome of PREVIOUS turn
    │
    ├─ accepted → collector.mark_feedback(prev_record_id, "positive",
    │                                      source="outcome_tracker")
    ├─ rejected → collector.mark_feedback(prev_record_id, "negative",
    │                                      source="outcome_tracker")
    └─ neutral  → no call (no signal)
```

This requires:
- `OutcomeTracker.get_last_outcome()` — new getter for last inferred outcome
- Tracking `last_record_id` in the chat loop (record returned by
  `capture_prompt_response`)

### Single-shot (ask command)

In `ask`, there is only one turn. OutcomeTracker closes the session with
`neutral` (no follow-up to compare against). No mark_feedback call — this
is correct behavior, not a gap.

## Changes by File

### `src/flywheel/collector.py`

| Change | Description |
|--------|-------------|
| `FeedbackRecord` dataclass | New, after InteractionRecord |
| `__init__` | Create `feedback/` directory |
| `mark_feedback()` | Add `source` param, append to sidecar when not in buffer |
| `_load_feedback_index()` | New private method, reads feedback/*.jsonl into dict |
| `export_training_pairs()` | Merge feedback_index before filtering |

### `src/flywheel/outcome_tracker.py`

| Change | Description |
|--------|-------------|
| `get_last_outcome()` | New method, returns last inferred OutcomeRecord or None |

### `src/interfaces/cli.py`

| Change | Description |
|--------|-------------|
| `chat()` | Bridge: after record_turn, check last outcome, call mark_feedback |
| `ask()` | No changes (single-shot = neutral = no feedback call) |

### `tests/test_flywheel.py`

| Test | Validates |
|------|-----------|
| `test_mark_feedback_persists_to_sidecar` | Capture → flush → mark_feedback → sidecar file has correct content |
| `test_mark_feedback_buffer_priority` | Record still in buffer → updates buffer, no sidecar write |
| `test_mark_feedback_source_field` | source="outcome_tracker" persisted correctly |
| `test_export_merges_feedback` | Capture → flush → mark_feedback → export → training pair has feedback |
| `test_feedback_last_wins` | Two feedbacks for same record_id → most recent wins |
| `test_load_feedback_index_empty` | No feedback files → empty dict, no crash |

### `tests/test_outcome_tracker.py`

| Test | Validates |
|------|-----------|
| `test_get_last_outcome` | Returns last inferred outcome after record_turn |

## Error Handling

- **Sidecar write fails** (disk full, read-only): try/except with
  `logger.warning`, do not crash. Feedback is lost but interaction continues.
  Same pattern as `capture-edits.sh` hook.
- **Corrupt sidecar line**: `_load_feedback_index()` skips invalid lines with
  `logger.warning`, same as `export_training_pairs` already does.
- **Orphan feedback** (record_id not in any interactions file): harmless,
  ignored during merge. mark_feedback is fire-and-forget.

## Out of Scope

- API/MCP integration with mark_feedback (future ticket if needed)
- Manual feedback UX in chat (OutcomeTracker covers this zero-friction)
- Retroactive feedback for historical interactions (no sidecar files exist)

## Alternatives Considered

### B: Rewrite JSONL in-place
Rejected. Violates append-only semantics, risks data corruption (lesson from
2026-02-06 incident where jq rewrite emptied 10 of 11 files).

### C: Delete mark_feedback entirely
Rejected. `export_training_pairs()` filters by `user_feedback` — without
mark_feedback, that filter never activates. The mechanism has functional value
when connected to OutcomeTracker.
