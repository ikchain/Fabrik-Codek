# Adaptive Forgetting Curve Design

## Problem

The current temporal decay uses a fixed half-life (90 days) for all edges regardless of how frequently they have been reinforced. An edge seen once and an edge reinforced 20 times both decay at the same rate. This is suboptimal: frequently reinforced knowledge represents stable, core competencies that should persist longer.

## Solution

Replace the fixed half-life with an **adaptive half-life** that increases with reinforcement count, inspired by spaced repetition research (Ebbinghaus forgetting curve + FOREVER/LECTOR papers).

### Formula

```
effective_half_life = base_half_life * (1 + REINFORCEMENT_FACTOR * reinforcement_count) ^ REINFORCEMENT_EXPONENT
weight = base_weight * 0.5 ^ (days_elapsed / effective_half_life)
```

Where:
- `base_half_life` = 90 days (existing default, configurable)
- `REINFORCEMENT_FACTOR` = 0.3 (how much each reinforcement extends half-life)
- `REINFORCEMENT_EXPONENT` = 0.5 (sublinear growth — diminishing returns)
- `reinforcement_count` = number of times the edge was reinforced via `add_relation()`

### Effective Half-Life Examples

| Reinforcements | Effective Half-Life | Relative to Base |
|---------------|-------------------|-----------------|
| 0 | 90.0 days | 1.0x |
| 1 | 102.5 days | 1.14x |
| 3 | 117.3 days | 1.30x |
| 5 | 129.0 days | 1.43x |
| 10 | 152.3 days | 1.69x |
| 20 | 186.5 days | 2.07x |
| 50 | 269.1 days | 2.99x |

The sublinear exponent (0.5 = square root) prevents runaway growth: even 50 reinforcements only ~3x the half-life.

### References

- FOREVER: Forgetting Curve-Inspired Memory Replay (Jan 2026): https://arxiv.org/abs/2601.03938
- Human-like Forgetting Curves in DNNs (Jun 2025): https://arxiv.org/abs/2506.12034
- LECTOR: Adaptive Spaced Learning (Aug 2025): https://arxiv.org/abs/2508.03275

## Changes

### 1. `add_relation()` — Track reinforcement count

When an edge is reinforced (already exists), increment `reinforcement_count`:

```python
# In the "edge exists" branch:
edge_meta["reinforcement_count"] = edge_meta.get("reinforcement_count", 0) + 1
```

New edges start with `reinforcement_count = 0`.

### 2. `apply_decay()` — Adaptive half-life

Replace fixed `half_life_days` with per-edge computation:

```python
reinforcement_count = meta.get("reinforcement_count", 0)
effective_half_life = half_life_days * (1 + REINFORCEMENT_FACTOR * reinforcement_count) ** REINFORCEMENT_EXPONENT
decay_factor = 0.5 ** (days_elapsed / effective_half_life)
```

### 3. Constants

```python
REINFORCEMENT_FACTOR: float = 0.3
REINFORCEMENT_EXPONENT: float = 0.5
```

Module-level constants in `graph_engine.py`, near the existing decay code.

### 4. Return stats

Add `reinforcement_count` stats to the `apply_decay()` return dict:
- `avg_reinforcement_count`: average across all decayed edges
- `max_reinforcement_count`: maximum seen

## Backward Compatibility

- Edges without `reinforcement_count` default to 0 → identical to current fixed decay
- No changes to the CLI interface (`fabrik graph decay` works as before)
- The `--half-life` flag still sets the base half-life
- `apply_decay()` signature unchanged (same args, same return keys + 2 new)

## Files Changed

| File | Change | Lines |
|------|--------|-------|
| `src/knowledge/graph_engine.py` | reinforcement_count in add_relation, adaptive formula in apply_decay | ~15 |
| `tests/test_graph_engine.py` | New tests for adaptive decay | ~80 |
| **Total** | | **~95** |
