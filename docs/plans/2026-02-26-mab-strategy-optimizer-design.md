# MAB Strategy Optimizer Design

## Problem

The current `StrategyOptimizer` uses fixed thresholds to adjust retrieval strategies:
- Acceptance rate >= 0.7: keep defaults
- Acceptance rate 0.5-0.7: mild boost (depth +1, graph_weight +0.1)
- Acceptance rate < 0.5: strong boost (depth +2, graph_weight +0.2, fulltext 0.1)

This has three limitations:
1. Cannot discover intermediate or novel configurations
2. Cannot learn that vector-focused strategies work better for certain contexts
3. Thresholds are arbitrary and don't adapt to the distribution of outcomes

## Solution

Replace fixed thresholds with **Thompson Sampling Multi-Armed Bandit** per (task_type, topic) context. Each context has 4 discrete arms representing retrieval configurations. Arms use Beta distributions updated with binary rewards (accepted=1, rejected=0).

### References

- Multi-Armed Bandits Meet LLMs (Survey, Jan 2026): https://arxiv.org/abs/2601.12945
- MAB with Surrogate Rewards (Jun 2025): https://arxiv.org/abs/2506.16658
- Adaptive Exploration for Latent-State Bandits: https://arxiv.org/abs/2602.05139

## Architecture

### Arms

4 arms defined relative to the task type's base strategy from `TASK_STRATEGIES`:

| Arm ID | Modifications vs base | Beta Prior | Maps to old system |
|--------|----------------------|------------|-------------------|
| `default` | None | Beta(2, 1) | rate >= 0.7 |
| `graph_boost` | graph_depth +1, graph_weight +0.1 | Beta(1, 1) | rate 0.5-0.7 |
| `deep_graph` | graph_depth +2, graph_weight +0.2, fulltext_weight 0.1 | Beta(1, 1) | rate < 0.5 |
| `vector_focus` | vector_weight +0.1, graph_weight -0.1 | Beta(1, 1) | NEW |

The `default` arm gets a warm start prior Beta(2,1) (expected value ~0.67) so the system initially prefers the known-good configuration. Other arms start with Beta(1,1) (uniform prior) for unbiased exploration.

### Selection (Thompson Sampling)

```python
def select_arm(context_key: str) -> str:
    arms = self._state[context_key]["arms"]
    samples = {
        arm_id: random.betavariate(arm["alpha"], arm["beta"])
        for arm_id, arm in arms.items()
    }
    return max(samples, key=samples.get)
```

### Update

```python
def update(context_key: str, arm_id: str, reward: float) -> None:
    arm = self._state[context_key]["arms"][arm_id]
    if reward > 0:
        arm["alpha"] += 1
    else:
        arm["beta"] += 1
    self._state[context_key]["total_pulls"] += 1
    self._dirty = True
```

### Integration Flow

```
TaskRouter.route(query)
    │
    ├─ 1. classify task_type (keywords/LLM)
    ├─ 2. detect topic
    ├─ 3. get competence_level
    ├─ 4. select model
    ├─ 5. mab.select_arm(task_type, topic)  ← NEW: Thompson sample
    │     returns (arm_id, RetrievalStrategy)
    ├─ 6. build system_prompt
    └─ return RoutingDecision (with arm_id in strategy)

OutcomeTracker.record_turn(query, response, decision, latency)
    │
    └─ stores arm_id in strategy dict (already stores full strategy)

CLI chat loop (after outcome inference)
    │
    ├─ if outcome != neutral:
    │     mab.update(context, arm_id, reward)
    └─ mab.save_state()  (lazy, only if dirty)
```

### Persistence

File: `data/profile/mab_state.json`

```json
{
  "debugging_postgresql": {
    "arms": {
      "default": {"alpha": 8, "beta": 3},
      "graph_boost": {"alpha": 3, "beta": 2},
      "deep_graph": {"alpha": 1, "beta": 4},
      "vector_focus": {"alpha": 2, "beta": 1}
    },
    "total_pulls": 24,
    "last_updated": "2026-02-26T12:00:00"
  },
  "architecture_fastapi": {
    "arms": {
      "default": {"alpha": 2, "beta": 1},
      "graph_boost": {"alpha": 5, "beta": 1},
      "deep_graph": {"alpha": 1, "beta": 1},
      "vector_focus": {"alpha": 1, "beta": 2}
    },
    "total_pulls": 13,
    "last_updated": "2026-02-26T14:30:00"
  }
}
```

### Backward Compatibility

- `strategy_overrides.json` continues to be generated as an "export" of best arms
- `compute_overrides()` method still works: exports the highest expected-value arm per context
- `fabrik competence build` continues generating overrides (now from MAB state)
- If `mab_state.json` doesn't exist, system starts with priors (works from day 0)
- TaskRouter still supports static overrides as fallback

## Design Decisions

1. **Context = (task_type, topic)** — Same as current overrides key format. Topic captures domain specificity; competence_level already influences model selection separately.

2. **4 discrete arms** — Not continuous parameters. With <100 outcomes per context, continuous exploration would be noisy. 4 arms give sufficient coverage of useful configurations.

3. **Beta distribution** — Natural conjugate prior for binary outcomes (accepted/rejected). Updates are O(1), no matrix operations needed.

4. **`random.betavariate` from stdlib** — No numpy dependency needed for 4 arms. Keeps the project lightweight.

5. **Neutral outcomes ignored** — Don't update any arm. Consistent with current system that only counts accepted/rejected.

6. **Warm start for `default` arm** — Beta(2,1) gives ~67% expected value, ensuring the system initially prefers known-good defaults. Exploration happens naturally as Thompson Sampling occasionally samples other arms higher.

7. **Lazy persistence** — Only write `mab_state.json` when state has changed (`_dirty` flag). Avoids unnecessary disk I/O on every turn.

8. **Arms relative to base** — Arm configs are computed from `TASK_STRATEGIES`, not hardcoded. If base strategies change, arms adapt automatically.

## Files Changed

| File | Change | Lines |
|------|--------|-------|
| `src/core/strategy_optimizer.py` | Rewrite: `MABStrategyOptimizer` class | ~130 |
| `src/core/task_router.py` | `route()` step 5 uses `mab.select_arm()` | ~20 |
| `src/flywheel/outcome_tracker.py` | Store `arm_id` in strategy dict | ~5 |
| `src/interfaces/cli.py` | Connect MAB update in chat/ask loops | ~15 |
| `tests/test_strategy_optimizer.py` | Rewrite + extend for MAB | ~30 new |
| **Total** | | **~200** |

## Testing Strategy

1. **Unit tests for MAB mechanics**: select_arm returns valid arm, update increments correct params
2. **Thompson Sampling convergence**: after many updates with one dominant arm, selection should prefer it (>80% of the time)
3. **Cold start**: new context with no history returns valid strategy using priors
4. **Persistence round-trip**: save_state + load_state preserves all Beta params
5. **Backward compat**: compute_overrides() produces valid strategy_overrides.json from MAB state
6. **Integration**: TaskRouter correctly uses MAB-selected strategy in routing decisions
