# SPRInG Incremental Profile Design

## Problem

`ProfileBuilder.build()` reconstructs the entire profile from scratch every time:
1. Scans ALL training pairs + auto-captures (~12k+ files)
2. Recomputes topic_weights, patterns, task_types from zero
3. Overwrites personal_profile.json completely

This is slow for large datalakes and loses trend information — it can't tell if the user is *migrating* from Angular to React.

## Solution

Add **incremental build mode** inspired by SPRInG (Selective Parametric Adaptation). Three components:

1. **Timestamp filtering**: Only analyze entries since `last_build_timestamp`
2. **Drift detection**: Cosine distance between current and new topic distributions
3. **EMA merge**: Blend new data into existing profile with adaptive alpha

### References

- SPRInG: Continual LLM Personalization (Jan 2026): https://arxiv.org/abs/2601.09974
- PESO: Continual Low-Rank Adapters (Oct 2025): https://arxiv.org/abs/2510.25093

## Incremental Build Flow

```
fabrik profile build --incremental
  │
  ├── 1. Load existing profile + last_build_timestamp
  │
  ├── 2. Filter: DatalakeAnalyzer processes only entries since timestamp
  │       (training pairs by file date, auto-captures by record timestamp)
  │
  ├── 3. Build incremental profile from new entries only
  │
  ├── 4. Detect drift: cosine distance(existing_topics, new_topics)
  │       drift_detected = distance > DRIFT_THRESHOLD (default 0.3)
  │
  ├── 5. Merge with EMA:
  │       alpha = 0.7 if drift_detected else 0.3
  │       merged_weight = alpha * new + (1 - alpha) * existing
  │
  ├── 6. Update replay buffer (top-20 most informative new entries)
  │
  └── 7. Save merged profile + update last_build_timestamp + append drift_history
```

## Changes

### `personal_profile.py` — ~120 lines added

**1. New metadata fields on PersonalProfile**

```python
@dataclass
class PersonalProfile:
    # ... existing fields ...
    last_build_timestamp: str | None = None
    build_mode: str = "full"
    drift_history: list[dict] = field(default_factory=list)
    replay_buffer: list[dict] = field(default_factory=list)
```

Backward compatible: `from_dict()` defaults missing keys, `to_dict()` includes them.

**2. `DatalakeAnalyzer.analyze_training_pairs(since=None)`** — add optional timestamp filter

When `since` is provided, only read JSONL files modified after that timestamp (by file mtime). This avoids reading old files entirely.

**3. `DatalakeAnalyzer.analyze_auto_captures(since=None)`** — same filter

For auto-captures, filter by file mtime AND by record-level `timestamp` field.

**4. `ProfileBuilder.build_incremental(output_path)` — new method (~60 lines)**

- Load existing profile from `output_path`
- If no existing profile or no `last_build_timestamp`: fall back to full `build()`
- Filter analysis to entries since `last_build_timestamp`
- If no new entries: return existing profile unchanged
- Build incremental profile from new entries
- Call `_detect_drift()` to check for topic distribution shift
- Call `_merge_profiles()` to EMA-blend
- Update `last_build_timestamp`, `build_mode`, `drift_history`
- Update `replay_buffer` with most informative new entries
- Save and return

**5. `ProfileBuilder._detect_drift(existing, incremental)` — new method (~20 lines)**

- Extract topic weight vectors from both profiles (aligned by topic union)
- Compute cosine distance: `1 - dot(a, b) / (norm(a) * norm(b))`
- Return `(drift_detected: bool, distance: float, topics_drifted: list[str])`
- `topics_drifted` = topics where |new_weight - old_weight| > 0.1

**6. `ProfileBuilder._merge_profiles(existing, incremental, alpha)` — new method (~30 lines)**

- EMA merge on `top_topics`: `merged = alpha * new + (1 - alpha) * existing`
- Union of topics (new topics appear, old topics decay)
- Re-sort by weight, keep top 10
- Merge `patterns`: keep existing, add any new ones not already present
- Merge `task_types_detected`: union, re-order by frequency
- `total_entries`: existing + incremental new entries
- `domain` and `domain_confidence`: keep existing (stable signal)

**7. `_compute_replay_buffer(new_entries, existing_topics, max_size=20)` — new function**

- Score each new entry by "novelty": topics/categories NOT in existing top_topics
- Keep top-20 by novelty score
- Store as list of `{"source": ..., "category": ..., "timestamp": ..., "novelty_score": ...}`

**8. Constants**

```python
DRIFT_THRESHOLD: float = 0.3      # cosine distance threshold
ALPHA_NORMAL: float = 0.3         # EMA alpha for gradual change
ALPHA_DRIFT: float = 0.7          # EMA alpha when drift detected
REPLAY_BUFFER_SIZE: int = 20      # max replay buffer entries
```

### `cli.py` — ~30 lines added

**1. `fabrik profile build --incremental` flag**

Add `--incremental` flag to existing `profile build` command. When set, calls `build_incremental()` instead of `build()`.

**2. `fabrik profile drift` subcommand**

New action "drift" that shows drift_history from the profile:
- Date, topics drifted, magnitude
- If empty: "No drift history yet. Run: fabrik profile build --incremental"

### No changes to:
- `save_profile()` / `load_profile()` — work as-is (dict-based, extra keys pass through)
- `get_active_profile()` — works as-is
- `to_system_prompt()` — works as-is
- Full `build()` — unchanged, still works exactly the same
- `CompetenceBuilder` — independent, reads datalake directly
- `TaskRouter` — reads profile via `get_active_profile()`, agnostic to build mode

## Files Changed

| File | Change | Lines |
|------|--------|-------|
| `src/core/personal_profile.py` | New fields, build_incremental, drift, merge, replay | ~120 |
| `src/interfaces/cli.py` | --incremental flag, drift action | ~30 |
| `tests/test_personal_profile.py` | New tests | ~150 |
| **Total** | | **~300** |

## Backward Compatibility

- Full `build()` unchanged — same args, same return, same behavior
- PersonalProfile new fields default to None/empty — old profiles load fine
- `to_dict()` adds new keys — consumers that don't know them ignore them
- CLI `fabrik profile build` (without --incremental) = same as before
- `fabrik profile show` shows new fields only if present
