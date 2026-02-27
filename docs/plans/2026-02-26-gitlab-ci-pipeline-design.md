# GitLab CI Pipeline

**Date:** 2026-02-26
**Status:** Approved

## Problem

Tests, linting, and format checks run only locally. No automated validation
on push or merge requests. Regressions can slip into main undetected.

## Design Principles

1. **Fail-fast** — Quality checks (seconds) block expensive test runs (minutes).
2. **Cache aggressively** — pip cache per branch avoids redundant downloads.
3. **Minimal config** — Tests mock all external services; CI needs no Ollama,
   no datalake, no GPU.
4. **Match production** — Use `python:3.12-slim` to match the actual dev
   environment (Python 3.12.3).

## Solution: 2-Stage Pipeline

### Architecture

```
push / MR
    │
    ├─ stage: quality
    │   └─ job: lint
    │       ├─ ruff check src/ tests/
    │       └─ black --check src/ tests/
    │
    └─ stage: test (runs only if quality passes)
        └─ job: test
            ├─ pytest --cov=src tests/
            └─ coverage regex for GitLab badge
```

### Jobs

| Job | Stage | Installs | Runs |
|-----|-------|----------|------|
| `lint` | quality | `pip install ruff black` | `ruff check src/ tests/` + `black --check src/ tests/` |
| `test` | test | `pip install -e .[dev]` | `pytest --cov=src --cov-report=term tests/` |

### Cache

```yaml
cache:
  key: ${CI_COMMIT_REF_SLUG}
  paths:
    - .cache/pip
```

### Triggers

- Push to `main`
- Merge requests

### Environment Variables

Tests mock all external services. Minimal variables prevent pydantic-settings
import errors:

```yaml
variables:
  FABRIK_OLLAMA_HOST: "http://localhost:11434"
  FABRIK_DEFAULT_MODEL: "qwen2.5-coder:7b"
```

### Coverage

GitLab extracts coverage from pytest output via regex:

```yaml
coverage: '/(?i)total.*? (100(?:\.0+)?\%|[1-9]?\d(?:\.\d+)?\%)$/'
```

This enables the native GitLab coverage badge without extra tools.

## Changes by File

### `.gitlab-ci.yml` (new)

| Change | Description |
|--------|-------------|
| `lint` job | ruff + black check in quality stage |
| `test` job | pytest + coverage in test stage |
| Cache config | pip cache per branch |
| Variables | Minimal env for pydantic-settings |

## Out of Scope

- GitHub Actions (repo publico usa GitHub, pero CI es para GitLab interno)
- Deploy automatico (no hay deploy target)
- Matrix multi-Python (un solo target: 3.12)
- mypy / type checking (no configurado actualmente)
- Badge en README (GitLab badge nativo ya funciona)
