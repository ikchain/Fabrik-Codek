# Hyper-Personalization Engine

**Date:** 2026-02-20
**Status:** Approved
**Branch:** TBD

## Vision

A 7B model that knows YOU is worth more than a 400B model that doesn't.

Fabrik-Codek is a **domain-agnostic cognitive architecture** that transforms a
generic 7B model into YOUR personal assistant. It learns from your interactions,
adapts to your domain, and gets better at YOUR specific tasks over time.

It works for any knowledge worker:
- A **developer** gets an assistant that knows their stack, patterns, and style
- A **lawyer** gets one that knows their case law, contract patterns, and legal style
- A **doctor** gets one that knows their protocols, prescriptions, and diagnostic patterns
- A **writer** gets one that knows their narrative style, vocabulary, and structure

The framework is domain-agnostic. The datalake IS the domain.

Open source pitch: "Your own 7B model that knows YOUR profession, learns YOUR
way of working, and gets better every day. Free, local, private."

## Problem

Today Fabrik has rich personal data (12,367 captures, 8,955 training pairs,
180 graph entities, 2,565 edges) but uses it generically:
- RAG retrieval is the same regardless of task type
- No awareness of what domains the system is competent in
- No tracking of whether responses were actually useful
- No learned user profile (preferences, style, patterns)
- Task types are hardcoded for code — doesn't adapt to other domains

The cognitive architecture exists but doesn't leverage its main advantage:
being PERSONAL.

## Design Principles

1. **Domain-agnostic by default** — No hardcoded task types, domains, or categories.
   Everything is learned from the user's datalake.
2. **Code-first reference implementation** — The developer use case is the first
   and most complete implementation, but the architecture never assumes "code".
3. **Zero configuration** — Profile, competence, and task types are all learned.
   The user just works; the system adapts.
4. **Datalake IS the domain** — Whatever data the user feeds becomes the domain.
   Legal documents → legal assistant. Medical notes → medical assistant.

## Design

### Component 1: Personal Profile

**File:** `src/core/personal_profile.py`

A profile LEARNED from the datalake, never manually configured.

**What it captures (domain-agnostic):**
- **Domain distribution**: What topics appear most (tech stack for devs,
  legal areas for lawyers, specialties for doctors)
- **Style patterns**: How the user writes/communicates (formal vs casual,
  verbose vs concise, language preferences)
- **Workflow patterns**: What tasks they do most, in what order, at what times
- **Terminology**: Domain-specific vocabulary the user uses frequently
- **Preferences**: Recurring choices and decisions from the datalake

**For developers specifically (auto-detected):**
- Tech stack distribution (languages, frameworks, databases)
- Coding patterns (async/sync, composition/inheritance, error handling)
- Naming conventions per language
- Testing preferences

**How it builds:**
- Analyze datalake content: extract categories, topics, patterns from any
  structured data (JSONL, markdown, text)
- Analyze graph: entity frequencies, relation patterns, domain clusters
- Use LLM (the local model itself) to summarize patterns found
- Periodic rebuild — runs as `fabrik profile build`

**Output:** JSON profile stored in `data/profile/personal_profile.json`

```json
{
  "domain": "software_development",
  "domain_confidence": 0.95,
  "top_topics": [
    {"topic": "postgresql", "weight": 0.18},
    {"topic": "fastapi", "weight": 0.15},
    {"topic": "typescript", "weight": 0.12}
  ],
  "style": {
    "formality": 0.6,
    "verbosity": 0.3,
    "language": "es"
  },
  "patterns": [
    "Prefers composition over inheritance",
    "Always adds error handling for external APIs",
    "Uses async/await over callbacks"
  ],
  "task_types_detected": [
    "code_review", "debugging", "architecture_decision",
    "explanation", "generation", "refactoring"
  ]
}
```

**Usage:** Injected into system prompt for LLM calls. "You are assisting a
[domain] professional who [patterns]. Adapt your responses accordingly."

### Component 2: Competence Model

**File:** `src/core/competence_model.py`

Maps domains/topics to competence levels based on available knowledge.
Domain-agnostic: works for any topic the datalake contains.

**How it works:**
- Count datalake entries per topic/category (training pairs, captures, etc.)
- Count graph entities and edge density per topic cluster
- Factor in recency (recent activity = higher weight)
- Compute competence score: `log(entries + 1) * entity_density * recency_weight`

**Competence levels:**
- **Expert** (score > 0.8): 50+ entries, dense graph cluster, recent activity
- **Competent** (0.4-0.8): 10-50 entries, some graph presence
- **Novice** (0.1-0.4): 1-10 entries, sparse graph
- **Unknown** (< 0.1): No data

**Behavior changes (domain-agnostic):**
- Expert: confident responses, use personal patterns from profile
- Competent: responses with RAG context, reference relevant personal experience
- Novice: honest disclosure ("I have limited context about your [topic] patterns")
- Unknown: suggest the user provide more data or use a larger model

**Output:** `data/profile/competence_map.json` - topic -> score mapping

### Component 3: Adaptive Task Pipeline

**File:** `src/core/task_router.py`

Different retrieval and prompting strategies per task type.
Task types are **learned from the datalake**, not hardcoded.

**Task type discovery:**
During `fabrik profile build`, the system clusters datalake entries by intent
and discovers what task types the user actually performs. For a developer this
might produce [code_review, debugging, explanation, generation]. For a lawyer
it might produce [contract_review, legal_research, case_analysis, drafting].

**Task recognition at query time:**
Classify incoming query into learned task types using:
1. Keyword matching against learned vocabulary per task type
2. LLM classification fallback (fast, uses local model)

**Per-task retrieval strategies (learned):**
Each task type builds a retrieval profile from outcome data:
- What RAG query formulation works best for this task type
- Optimal graph traversal depth
- What extra context improves responses (profile data, recent entries, etc.)

Initial defaults (before outcome data exists):

| Strategy | RAG Approach | Graph Depth | Extra Context |
|----------|-------------|-------------|---------------|
| analysis | Full document | 2 | Related concepts |
| review | Source + reference | 1 | Style/patterns from profile |
| problem_solving | Problem description | 2 | Similar past problems |
| creation | Requirements | 1 | Personal patterns |
| explanation | Concept terms | 3 | Related concepts chain |

**Per-task system prompts:**
Each task type gets a different system prompt that includes:
- Task-specific instructions adapted to the domain
- Relevant personal profile data
- Competence-aware framing (confident vs cautious)

### Component 4: Outcome Tracking

**File:** `src/flywheel/outcome_tracker.py`

Close the cognitive loop: know if the response was useful.

**Signals captured:**
- `accepted`: User didn't modify the response (positive signal)
- `edited`: User took the response but modified it (improvement signal)
- `rejected`: User asked again or ignored (negative signal)
- `escalated`: User went to a larger model for this (competence gap signal)

**How to capture:**
- In `fabrik chat`: track if user says "thanks"/follow-up (accepted),
  or rephrases the same question (rejected)
- In MCP server: track if the tool result was used by the host agent
- In API: track response usage patterns

**Feedback storage:** `datalake/01-raw/outcomes/YYYY-MM-DD_outcomes.jsonl`

**Feedback loop:**
- Aggregate outcomes per task_type + topic
- Adjust competence model scores based on acceptance rates
- Refine retrieval strategies: "For [task_type] in [topic], strategy X
  had 85% acceptance"
- Over time, the system learns WHICH retrieval approach works best
  for WHICH task type in WHICH domain — fully adaptive

## Architecture

```
User Query
    |
    v
[Task Recognition] --> task_type (learned from datalake)
    |
    v
[Competence Check] --> competence_level for detected topic
    |
    v
[Adaptive Retrieval] --> strategy selected per task_type
    |                     + personal profile context
    v
[LLM with adapted system prompt]
    |
    v
[Response + confidence indicator]
    |
    v
[Outcome Tracking] --> feeds back to competence model
                       + refines retrieval strategies
```

## New Files

- `src/core/personal_profile.py` - Profile builder and loader
- `src/core/competence_model.py` - Topic competence scoring
- `src/core/task_router.py` - Task classification and adaptive strategy selection
- `src/flywheel/outcome_tracker.py` - Response outcome tracking
- `data/profile/` - Generated profile and competence map
- `tests/test_personal_profile.py`
- `tests/test_competence_model.py`
- `tests/test_task_router.py`
- `tests/test_outcome_tracker.py`

## Modified Files

- `src/interfaces/cli.py` - Wire adaptive pipeline into ask/chat commands
- `src/interfaces/api.py` - Wire adaptive pipeline into API endpoints
- `src/interfaces/mcp_server.py` - Wire adaptive pipeline into MCP tools
- `src/core/llm_client.py` - Accept task-aware system prompts

## CLI Commands

- `fabrik profile build` - Build/rebuild personal profile from datalake
- `fabrik profile show` - Show current profile (domains, patterns, competence)
- `fabrik competence` - Show competence map by topic

## Implementation Order

1. **Personal Profile** - Foundation, needed by everything else
2. **Competence Model** - Depends on profile data
3. **Task Router** - Uses profile + competence to adapt behavior
4. **Outcome Tracking** - Closes the loop, depends on router being in place

## Success Criteria

- `fabrik profile build` discovers domain and task types without configuration
- `fabrik profile show` displays accurate topics and preferences
- `fabrik competence` shows realistic scores per topic
- Different task types produce visibly different retrieval strategies
- System works identically whether datalake contains code, legal docs, or medical notes
- Acceptance rate is trackable and trends upward over time

## Example: Developer (current use case)

```
$ fabrik profile show
Domain: Software Development (confidence: 0.95)
Top topics: PostgreSQL (18%), FastAPI (15%), TypeScript (12%)
Patterns: async/await, composition, hexagonal architecture
Task types: code_review, debugging, explanation, generation, refactoring
Competence: PostgreSQL (Expert), Angular (Competent), Kubernetes (Novice)
```

## Example: Lawyer (hypothetical)

```
$ fabrik profile show
Domain: Legal Practice (confidence: 0.92)
Top topics: Civil Law (22%), Contract Law (18%), Labor Law (14%)
Patterns: formal style, Art. references, case-based reasoning
Task types: contract_review, legal_research, case_analysis, drafting
Competence: Civil Law (Expert), Tax Law (Novice), Criminal Law (Unknown)
```
