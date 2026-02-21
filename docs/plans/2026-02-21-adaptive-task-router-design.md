# Adaptive Task Router — Design

## Problem

`fabrik ask` and `fabrik chat` always use the same model, same retrieval strategy, and same system prompt regardless of task type or user competence. A debugging question about PostgreSQL (where the user is Expert) gets the same treatment as an architecture question about Kubernetes (where the user is Novice).

## Solution

A `TaskRouter` module that classifies incoming queries and produces a `RoutingDecision` containing: task type, detected topic, competence level, model selection, retrieval strategy, and an adapted system prompt.

## Data Model

```python
@dataclass
class RetrievalStrategy:
    use_rag: bool = True
    use_graph: bool = True
    graph_depth: int = 2
    vector_weight: float = 0.6
    graph_weight: float = 0.4
    fulltext_weight: float = 0.0

@dataclass
class RoutingDecision:
    task_type: str              # "debugging", "code_review", "architecture", etc.
    topic: str | None           # Detected topic from competence map, or None
    competence_level: str       # "Expert", "Competent", "Novice", "Unknown"
    model: str                  # default_model or fallback_model
    strategy: RetrievalStrategy
    system_prompt: str          # Profile + competence + task-specific instructions
    classification_method: str  # "keyword" or "llm"
```

## Classification: Hybrid (Keywords + LLM Fallback)

### Step 1: Keyword Matching

Static vocabulary per task type, learned from datalake categories:

| Task Type | Keywords |
|-----------|----------|
| debugging | error, bug, fix, crash, traceback, exception, fails, broken |
| code_review | review, refactor, clean, improve, quality, smell |
| architecture | design, pattern, structure, ddd, hexagonal, module |
| explanation | explain, how, why, what is, difference, compare |
| testing | test, assert, mock, coverage, pytest, spec |
| devops | deploy, docker, kubernetes, ci/cd, pipeline, terraform |
| ml_engineering | model, training, fine-tune, embedding, rag, llm |

Scoring: count normalized matches per task type. If top score > 0.3 threshold, use that type. Otherwise, fall back to LLM.

### Step 2: LLM Fallback

Short classification prompt to Qwen (~0.5s):
```
Classify this query into ONE task type: debugging, code_review, architecture,
explanation, testing, devops, ml_engineering, general.
Query: "{query}"
Answer with just the task type.
```

Only invoked when keyword matching yields low confidence.

### Topic Detection

Direct case-insensitive token matching between query words and `competence_map.topics[].topic`. No stemming needed — topics are already keywords.

## Retrieval Strategies (Static)

| Task Type | graph_depth | vector_weight | graph_weight | Rationale |
|-----------|-------------|---------------|--------------|-----------|
| debugging | 2 | 0.5 | 0.5 | Errors have causal relationships in graph |
| code_review | 1 | 0.7 | 0.3 | Pattern matching is vector-heavy |
| architecture | 3 | 0.4 | 0.6 | Deep component relationships |
| explanation | 2 | 0.6 | 0.4 | Balanced |
| testing | 2 | 0.6 | 0.4 | Balanced |
| devops | 1 | 0.7 | 0.3 | Specific commands, vector-heavy |
| ml_engineering | 2 | 0.5 | 0.5 | Balanced |
| general | 2 | 0.6 | 0.4 | Defaults |

## Escalation Logic

- `competence_level in ("Novice", "Unknown")` → `model = settings.fallback_model`
- `competence_level in ("Expert", "Competent")` → `model = settings.default_model`

## System Prompt Construction (3 Layers)

1. **Base**: `profile.to_system_prompt()` — personal profile
2. **Competence**: `competence_map.to_system_prompt_fragment()` — expertise levels
3. **Task-specific** (new): instructions per task type

Task-specific instructions:

| Task Type | Instruction |
|-----------|-------------|
| debugging | "Focus on root cause analysis. Be direct about the fix." |
| code_review | "Be specific about issues. Reference patterns and best practices." |
| architecture | "Explain trade-offs. Consider scalability and maintainability." |
| explanation | "Be clear and structured. Use examples when helpful." |
| testing | "Focus on edge cases and coverage. Suggest test strategies." |
| devops | "Be precise with commands and configs. Warn about destructive ops." |
| ml_engineering | "Reference specific techniques. Distinguish theory from practice." |
| general | (no additional instruction) |

## Integration Points

### CLI (`cli.py`)
```python
router = TaskRouter(competence_map, profile, settings)
decision = await router.route(query)

# RAG with adapted strategy
context = await hybrid.query_with_context(query, graph_depth=decision.strategy.graph_depth)

# LLM with adapted model and prompt
response = await client.generate(final_prompt, system=decision.system_prompt, model=decision.model)
```

Same pattern in `ask()` and `chat()` commands.

### API (`api.py`)
Endpoint `/ask` uses the same `router.route()`.

### MCP (`mcp_server.py`)
Tool `fabrik_ask` uses the same `router.route()`.

### CLI Debug Command
New `fabrik router test "query"` command to inspect classification without executing:
```
$ fabrik router test "why is my postgres query slow?"
Task Type:   debugging (keyword, confidence: 0.67)
Topic:       postgresql (Expert, score: 0.85)
Model:       qwen2.5-coder:14b (no escalation)
Strategy:    graph_depth=2, vector=0.5, graph=0.5
```

## Files

| File | Change |
|------|--------|
| `src/core/task_router.py` | NEW — TaskRouter, RoutingDecision, RetrievalStrategy (~250-300 lines) |
| `src/interfaces/cli.py` | Use router in ask() and chat(), add `router` command |
| `src/interfaces/api.py` | Use router in /ask endpoint |
| `src/interfaces/mcp_server.py` | Use router in fabrik_ask tool |
| `tests/test_task_router.py` | NEW — ~60-70 tests |

## Tests (~60-70)

| Group | Count | Coverage |
|-------|-------|----------|
| Keyword classification | ~12 | Each task type + edge cases + no match |
| LLM fallback | ~6 | Invoked when keywords fail, response parsing |
| Topic detection | ~8 | Direct match, no match, case-insensitive |
| Escalation logic | ~8 | Expert→default, Novice→fallback, Unknown→fallback |
| Strategy selection | ~10 | Each task type returns correct strategy |
| System prompt construction | ~8 | 3 layers, task-specific instructions |
| Integration (route()) | ~8 | End-to-end: query → full decision |

## Future
Outcome Tracking will close the loop: track whether routing decisions led to good outcomes, and feed back to adjust strategies and escalation thresholds. The static strategies in this design are the starting point that Outcome Tracking will make adaptive.
