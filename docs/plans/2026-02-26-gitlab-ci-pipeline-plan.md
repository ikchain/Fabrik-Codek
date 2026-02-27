# GitLab CI Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `.gitlab-ci.yml` that runs ruff, black, and pytest on every push to main and merge request.

**Architecture:** Two-stage pipeline — quality (lint + format check) blocks test (pytest + coverage). Pip cache per branch. `python:3.12-slim` image. Minimal env vars for pydantic-settings.

**Tech Stack:** GitLab CI, Python 3.12, ruff, black, pytest, pytest-cov

---

### Task 1: Create `.gitlab-ci.yml`

**Files:**
- Create: `.gitlab-ci.yml`

**Context:**
- Design doc: `docs/plans/2026-02-26-gitlab-ci-pipeline-design.md`
- Existing tool config: `pyproject.toml` (ruff, black, pytest sections)
- No CI config exists yet in the project

**Step 1: Create the CI configuration file**

```yaml
image: python:3.12-slim

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"
  FABRIK_OLLAMA_HOST: "http://localhost:11434"
  FABRIK_DEFAULT_MODEL: "qwen2.5-coder:7b"

cache:
  key: ${CI_COMMIT_REF_SLUG}
  paths:
    - .cache/pip

stages:
  - quality
  - test

lint:
  stage: quality
  before_script:
    - pip install ruff black
  script:
    - ruff check src/ tests/
    - black --check src/ tests/

test:
  stage: test
  before_script:
    - pip install -e .[dev]
  script:
    - pytest --cov=src --cov-report=term tests/
  coverage: '/(?i)total.*? (100(?:\.0+)?\%|[1-9]?\d(?:\.\d+)?\%)$/'
```

**Step 2: Validate YAML syntax locally**

Run: `python -c "import yaml; yaml.safe_load(open('.gitlab-ci.yml'))"`
Expected: No error (valid YAML)

If `yaml` not installed: `python -c "import json, re; open('.gitlab-ci.yml').read()"` — at minimum verify the file is readable.

**Step 3: Verify ruff and black work locally with same commands**

Run: `ruff check src/ tests/ && black --check src/ tests/`
Expected: Both pass (if not, fix issues before committing)

**Step 4: Verify pytest works with coverage flag**

Run: `pytest --cov=src --cov-report=term tests/ -q 2>&1 | tail -5`
Expected: `828 passed` + coverage summary

**Step 5: Commit**

```bash
git add .gitlab-ci.yml docs/plans/2026-02-26-gitlab-ci-pipeline-design.md docs/plans/2026-02-26-gitlab-ci-pipeline-plan.md
git commit -m "FEAT: Add GitLab CI pipeline"
```

### Task 2: Push and verify pipeline runs

**Step 1: Push to GitLab**

```bash
git push origin main
```

**Step 2: Verify pipeline status**

Check the GitLab project at `https://your-gitlab-instance/project/-/pipelines` (or equivalent URL) to confirm:
- Pipeline triggered on push
- `lint` job passes (quality stage)
- `test` job passes (test stage)
- Coverage percentage extracted

**Step 3: If pipeline fails**

- **Runner not configured**: User needs to register a GitLab runner on the server
- **Dependency install fails**: Check if the runner has internet access
- **Tests fail**: Compare with local `pytest` output, check env vars
- **ruff/black fail**: Fix locally, commit, push again

### Task 3: Update CLAUDE.md and close ticket

**Step 1: Add CI info to CLAUDE.md**

Add to the "Comandos principales" section or create a new "CI/CD" section:
```
## CI/CD
- **GitLab CI**: `.gitlab-ci.yml` — 2 stages (quality → test)
- **Trigger**: push a main + merge requests
- **Jobs**: ruff + black (quality), pytest + coverage (test)
```

**Step 2: Commit docs update**

```bash
git add CLAUDE.md
git commit -m "DOCS: Update CLAUDE.md post "
```

**Step 3: Transition  to Done in Jira**
