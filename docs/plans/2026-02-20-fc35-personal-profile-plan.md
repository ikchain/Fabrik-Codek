# Personal Profile - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Personal Profile that is LEARNED from the datalake — never manually configured. It detects the user's domain, top topics, style patterns, and preferred task types automatically.

**Architecture:** A `ProfileBuilder` reads training pairs (categories/tags), auto-captures (projects/tools/file types), and graph entities (types/frequencies). It produces a JSON profile that gets injected into the LLM system prompt. Domain-agnostic: works for code, legal, medical, or any datalake content.

**Tech Stack:** Python 3.12, Pydantic models, async file I/O (aiofiles), existing datalake connector patterns, Typer CLI, Rich formatting, pytest with temp dirs.

---

### Task 1: Profile Data Model

**Files:**
- Create: `src/core/personal_profile.py`
- Test: `tests/test_personal_profile.py`

**Step 1: Write the failing test — Profile schema**

```python
"""Tests for Personal Profile."""

import json
import pytest
import tempfile
from pathlib import Path


class TestProfileSchema:
    """Test the profile data model."""

    def test_empty_profile_has_defaults(self):
        from src.core.personal_profile import PersonalProfile

        profile = PersonalProfile()
        assert profile.domain == "unknown"
        assert profile.domain_confidence == 0.0
        assert profile.top_topics == []
        assert profile.patterns == []
        assert profile.task_types_detected == []
        assert profile.style.formality == 0.5
        assert profile.total_entries == 0

    def test_profile_to_dict(self):
        from src.core.personal_profile import PersonalProfile, TopicWeight, StyleProfile

        profile = PersonalProfile(
            domain="software_development",
            domain_confidence=0.95,
            top_topics=[TopicWeight(topic="postgresql", weight=0.18)],
            style=StyleProfile(formality=0.6, verbosity=0.3, language="es"),
            patterns=["Prefers async/await"],
            task_types_detected=["debugging", "code_review"],
            total_entries=500,
        )
        d = profile.to_dict()
        assert d["domain"] == "software_development"
        assert d["top_topics"][0]["topic"] == "postgresql"
        assert d["style"]["language"] == "es"

    def test_profile_save_and_load(self, tmp_path):
        from src.core.personal_profile import PersonalProfile, TopicWeight, save_profile, load_profile

        profile = PersonalProfile(
            domain="legal_practice",
            domain_confidence=0.88,
            top_topics=[TopicWeight(topic="civil_law", weight=0.22)],
            patterns=["Cites Art. references"],
            total_entries=200,
        )
        filepath = tmp_path / "profile.json"
        save_profile(profile, filepath)
        loaded = load_profile(filepath)
        assert loaded.domain == "legal_practice"
        assert loaded.top_topics[0].topic == "civil_law"
        assert loaded.total_entries == 200

    def test_load_nonexistent_returns_empty(self, tmp_path):
        from src.core.personal_profile import load_profile

        loaded = load_profile(tmp_path / "nope.json")
        assert loaded.domain == "unknown"

    def test_profile_to_system_prompt(self):
        from src.core.personal_profile import PersonalProfile, TopicWeight

        profile = PersonalProfile(
            domain="software_development",
            domain_confidence=0.95,
            top_topics=[
                TopicWeight(topic="postgresql", weight=0.18),
                TopicWeight(topic="fastapi", weight=0.15),
            ],
            patterns=["Prefers async/await", "Uses hexagonal architecture"],
            task_types_detected=["debugging", "code_review"],
        )
        prompt = profile.to_system_prompt()
        assert "software_development" in prompt
        assert "postgresql" in prompt
        assert "async/await" in prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_empty_profile_gives_generic_prompt(self):
        from src.core.personal_profile import PersonalProfile

        profile = PersonalProfile()
        prompt = profile.to_system_prompt()
        assert "general" in prompt.lower() or len(prompt) < 200
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_personal_profile.py -v --tb=short 2>&1 | tail -20`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.core.personal_profile'`

**Step 3: Write minimal implementation**

```python
"""Personal Profile — learned from the datalake, never configured manually.

The profile captures domain, top topics, style patterns, and task types
detected from the user's data. Domain-agnostic: works for code, legal,
medical, or any datalake content.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import structlog

logger = structlog.get_logger()


@dataclass
class TopicWeight:
    """A topic with its relative weight in the profile."""

    topic: str
    weight: float = 0.0  # 0-1, relative frequency


@dataclass
class StyleProfile:
    """Communication style detected from the datalake."""

    formality: float = 0.5  # 0=casual, 1=formal
    verbosity: float = 0.5  # 0=concise, 1=verbose
    language: str = "en"  # Primary language detected


@dataclass
class PersonalProfile:
    """A user's personal profile, learned from their datalake."""

    domain: str = "unknown"
    domain_confidence: float = 0.0
    top_topics: list[TopicWeight] = field(default_factory=list)
    style: StyleProfile = field(default_factory=StyleProfile)
    patterns: list[str] = field(default_factory=list)
    task_types_detected: list[str] = field(default_factory=list)
    total_entries: int = 0
    built_at: str = ""

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return asdict(self)

    def to_system_prompt(self) -> str:
        """Generate a system prompt fragment from this profile."""
        if self.domain == "unknown" or not self.top_topics:
            return "You are a helpful assistant. Adapt to the user's needs."

        topics_str = ", ".join(
            f"{t.topic} ({t.weight:.0%})" for t in self.top_topics[:5]
        )
        patterns_str = "; ".join(self.patterns[:5]) if self.patterns else "none detected"

        return (
            f"You are assisting a {self.domain.replace('_', ' ')} professional. "
            f"Their top topics: {topics_str}. "
            f"Known patterns: {patterns_str}. "
            f"Adapt your responses to their domain expertise and style."
        )


def save_profile(profile: PersonalProfile, path: Path) -> None:
    """Save profile to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.to_dict(), indent=2, ensure_ascii=False))
    logger.info("profile_saved", path=str(path))


def load_profile(path: Path) -> PersonalProfile:
    """Load profile from JSON file. Returns empty profile if not found."""
    if not path.exists():
        return PersonalProfile()
    try:
        data = json.loads(path.read_text())
        style_data = data.pop("style", {})
        topics_data = data.pop("top_topics", [])
        return PersonalProfile(
            **data,
            style=StyleProfile(**style_data),
            top_topics=[TopicWeight(**t) for t in topics_data],
        )
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.warning("profile_load_error", error=str(e))
        return PersonalProfile()
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_personal_profile.py -v --tb=short 2>&1 | tail -20`
Expected: 6 passed

**Step 5: Commit**

```
FEAT: Add PersonalProfile data model
```

---

### Task 2: Datalake Analyzer

Analyzes training pairs and auto-captures to extract topic distribution, file types, projects, and language.

**Files:**
- Modify: `src/core/personal_profile.py`
- Test: `tests/test_personal_profile.py`

**Step 1: Write the failing test — Datalake analysis**

Add to `tests/test_personal_profile.py`:

```python
class TestDatalakeAnalyzer:
    """Test datalake content analysis."""

    @pytest.fixture
    def sample_training_pairs(self, tmp_path):
        """Create sample training pair JSONL files."""
        tp_dir = tmp_path / "02-processed" / "training-pairs"
        tp_dir.mkdir(parents=True)

        # File 1: postgresql pairs
        pairs1 = [
            {"instruction": "How to optimize a query?", "output": "Use EXPLAIN...", "category": "postgresql", "tags": ["postgresql", "performance"]},
            {"instruction": "Index types?", "output": "B-tree, hash...", "category": "postgresql", "tags": ["postgresql", "indexing"]},
            {"instruction": "Connection pooling?", "output": "Use pgbouncer...", "category": "postgresql", "tags": ["postgresql", "connections"]},
        ]
        (tp_dir / "postgresql-basics.jsonl").write_text(
            "\n".join(json.dumps(p) for p in pairs1)
        )

        # File 2: debugging pairs
        pairs2 = [
            {"instruction": "Fix timeout error", "output": "Add retry...", "category": "debugging", "tags": ["debugging", "timeout"]},
        ]
        (tp_dir / "debugging-basics.jsonl").write_text(
            "\n".join(json.dumps(p) for p in pairs2)
        )
        return tmp_path

    @pytest.fixture
    def sample_auto_captures(self, tmp_path):
        """Create sample auto-capture JSONL files."""
        ac_dir = tmp_path / "01-raw" / "code-changes"
        ac_dir.mkdir(parents=True)

        captures = [
            {"timestamp": "2026-02-20T10:00:00", "type": "auto_capture", "tool": "Edit", "project": "my-api", "file_modified": "/home/user/my-api/src/main.py", "change_type": "edit"},
            {"timestamp": "2026-02-20T10:05:00", "type": "auto_capture", "tool": "Write", "project": "my-api", "file_modified": "/home/user/my-api/tests/test_main.py", "change_type": "write"},
            {"timestamp": "2026-02-20T10:10:00", "type": "auto_capture", "tool": "Edit", "project": "frontend", "file_modified": "/home/user/frontend/src/App.tsx", "change_type": "edit"},
        ]
        (ac_dir / "2026-02-20_auto-captures.jsonl").write_text(
            "\n".join(json.dumps(c) for c in captures)
        )
        return tmp_path

    def test_analyze_training_pairs(self, sample_training_pairs):
        from src.core.personal_profile import DatalakeAnalyzer

        analyzer = DatalakeAnalyzer(datalake_path=sample_training_pairs)
        result = analyzer.analyze_training_pairs()
        assert result["total_pairs"] == 4
        assert "postgresql" in result["categories"]
        assert result["categories"]["postgresql"] == 3
        assert "debugging" in result["categories"]
        assert "postgresql" in result["tags"]

    def test_analyze_auto_captures(self, sample_auto_captures):
        from src.core.personal_profile import DatalakeAnalyzer

        analyzer = DatalakeAnalyzer(datalake_path=sample_auto_captures)
        result = analyzer.analyze_auto_captures()
        assert result["total_captures"] == 3
        assert "my-api" in result["projects"]
        assert result["projects"]["my-api"] == 2
        assert ".py" in result["file_extensions"]
        assert ".tsx" in result["file_extensions"]
        assert "Edit" in result["tools"]

    def test_analyze_empty_datalake(self, tmp_path):
        from src.core.personal_profile import DatalakeAnalyzer

        analyzer = DatalakeAnalyzer(datalake_path=tmp_path)
        tp = analyzer.analyze_training_pairs()
        ac = analyzer.analyze_auto_captures()
        assert tp["total_pairs"] == 0
        assert ac["total_captures"] == 0
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_personal_profile.py::TestDatalakeAnalyzer -v --tb=short`
Expected: FAIL — `ImportError: cannot import name 'DatalakeAnalyzer'`

**Step 3: Write minimal implementation**

Add to `src/core/personal_profile.py`:

```python
from collections import Counter


class DatalakeAnalyzer:
    """Analyzes datalake content to extract profile signals."""

    def __init__(self, datalake_path: Path):
        self.datalake_path = datalake_path

    def _read_jsonl(self, filepath: Path) -> list[dict]:
        """Read JSONL file, one JSON object per line."""
        records = []
        if not filepath.exists():
            return records
        for line in filepath.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    def analyze_training_pairs(self) -> dict:
        """Analyze training pairs for categories, tags, and language."""
        tp_dir = self.datalake_path / "02-processed" / "training-pairs"
        categories: Counter = Counter()
        tags: Counter = Counter()
        total = 0

        if not tp_dir.exists():
            return {"total_pairs": 0, "categories": {}, "tags": {}}

        for jsonl_file in sorted(tp_dir.glob("*.jsonl")):
            for record in self._read_jsonl(jsonl_file):
                total += 1
                if cat := record.get("category"):
                    categories[cat] += 1
                for tag in record.get("tags", []):
                    tags[tag] += 1

        return {
            "total_pairs": total,
            "categories": dict(categories),
            "tags": dict(tags),
        }

    def analyze_auto_captures(self) -> dict:
        """Analyze auto-captures for projects, file types, tools."""
        ac_dir = self.datalake_path / "01-raw" / "code-changes"
        projects: Counter = Counter()
        file_extensions: Counter = Counter()
        tools: Counter = Counter()
        total = 0

        if not ac_dir.exists():
            return {"total_captures": 0, "projects": {}, "file_extensions": {}, "tools": {}}

        for jsonl_file in sorted(ac_dir.glob("*auto-captures*.jsonl")):
            for record in self._read_jsonl(jsonl_file):
                total += 1
                if proj := record.get("project"):
                    projects[proj] += 1
                if fpath := record.get("file_modified"):
                    ext = Path(fpath).suffix
                    if ext:
                        file_extensions[ext] += 1
                if tool := record.get("tool"):
                    tools[tool] += 1

        return {
            "total_captures": total,
            "projects": dict(projects),
            "file_extensions": dict(file_extensions),
            "tools": dict(tools),
        }
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_personal_profile.py::TestDatalakeAnalyzer -v --tb=short`
Expected: 3 passed

**Step 5: Commit**

```
FEAT: Add DatalakeAnalyzer for profile signals
```

---

### Task 3: Domain Detection and Profile Builder

Combines datalake analysis + graph stats to detect domain, compute topic weights, and build the full profile.

**Files:**
- Modify: `src/core/personal_profile.py`
- Test: `tests/test_personal_profile.py`

**Step 1: Write the failing test — Profile builder**

Add to `tests/test_personal_profile.py`:

```python
class TestProfileBuilder:
    """Test the full profile build process."""

    @pytest.fixture
    def datalake_with_code(self, tmp_path):
        """Datalake that looks like a developer's."""
        tp_dir = tmp_path / "02-processed" / "training-pairs"
        tp_dir.mkdir(parents=True)
        ac_dir = tmp_path / "01-raw" / "code-changes"
        ac_dir.mkdir(parents=True)

        # 20 postgresql pairs, 10 debugging, 5 angular
        pairs = []
        for i in range(20):
            pairs.append({"instruction": f"pg query {i}", "output": "...", "category": "postgresql", "tags": ["postgresql"]})
        for i in range(10):
            pairs.append({"instruction": f"debug {i}", "output": "...", "category": "debugging", "tags": ["debugging"]})
        for i in range(5):
            pairs.append({"instruction": f"angular {i}", "output": "...", "category": "angular", "tags": ["angular"]})
        (tp_dir / "all.jsonl").write_text("\n".join(json.dumps(p) for p in pairs))

        # Auto-captures with .py and .ts files
        captures = []
        for i in range(15):
            captures.append({"timestamp": "2026-01-01T00:00:00", "type": "auto_capture", "tool": "Edit", "project": "backend", "file_modified": f"/src/file{i}.py", "change_type": "edit"})
        for i in range(5):
            captures.append({"timestamp": "2026-01-01T00:00:00", "type": "auto_capture", "tool": "Edit", "project": "frontend", "file_modified": f"/src/comp{i}.tsx", "change_type": "edit"})
        (ac_dir / "2026-01-01_auto-captures.jsonl").write_text("\n".join(json.dumps(c) for c in captures))

        return tmp_path

    @pytest.fixture
    def datalake_with_legal(self, tmp_path):
        """Datalake that looks like a lawyer's."""
        tp_dir = tmp_path / "02-processed" / "training-pairs"
        tp_dir.mkdir(parents=True)

        pairs = []
        for i in range(15):
            pairs.append({"instruction": f"consulta civil {i}", "output": "...", "category": "civil_law", "tags": ["civil_law", "contracts"]})
        for i in range(8):
            pairs.append({"instruction": f"caso laboral {i}", "output": "...", "category": "labor_law", "tags": ["labor_law"]})
        (tp_dir / "legal.jsonl").write_text("\n".join(json.dumps(p) for p in pairs))
        return tmp_path

    def test_build_developer_profile(self, datalake_with_code):
        from src.core.personal_profile import ProfileBuilder

        builder = ProfileBuilder(datalake_path=datalake_with_code)
        profile = builder.build()

        assert profile.domain == "software_development"
        assert profile.domain_confidence > 0.5
        assert profile.top_topics[0].topic == "postgresql"
        assert profile.total_entries > 0
        assert len(profile.task_types_detected) > 0

    def test_build_legal_profile(self, datalake_with_legal):
        from src.core.personal_profile import ProfileBuilder

        builder = ProfileBuilder(datalake_path=datalake_with_legal)
        profile = builder.build()

        assert profile.domain != "software_development"
        assert profile.top_topics[0].topic == "civil_law"
        assert profile.total_entries == 23

    def test_build_empty_datalake(self, tmp_path):
        from src.core.personal_profile import ProfileBuilder

        builder = ProfileBuilder(datalake_path=tmp_path)
        profile = builder.build()
        assert profile.domain == "unknown"
        assert profile.total_entries == 0

    def test_build_saves_profile(self, datalake_with_code, tmp_path):
        from src.core.personal_profile import ProfileBuilder, load_profile

        output = tmp_path / "profile.json"
        builder = ProfileBuilder(datalake_path=datalake_with_code)
        builder.build(output_path=output)

        loaded = load_profile(output)
        assert loaded.domain == "software_development"
        assert loaded.total_entries > 0

    def test_topic_weights_sum_to_roughly_one(self, datalake_with_code):
        from src.core.personal_profile import ProfileBuilder

        builder = ProfileBuilder(datalake_path=datalake_with_code)
        profile = builder.build()

        total_weight = sum(t.weight for t in profile.top_topics)
        assert 0.9 <= total_weight <= 1.1  # Allow small rounding
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_personal_profile.py::TestProfileBuilder -v --tb=short`
Expected: FAIL — `ImportError: cannot import name 'ProfileBuilder'`

**Step 3: Write minimal implementation**

Add to `src/core/personal_profile.py`:

```python
from datetime import datetime

# File extension → domain signal mapping
CODE_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx", ".java", ".go", ".rs", ".rb", ".cs", ".cpp", ".c", ".h"}
CODE_CATEGORIES = {"debugging", "testing", "code-review", "refactoring", "agents", "postgresql", "docker",
                   "terraform", "typescript", "angular", "ddd", "kubernetes"}


class ProfileBuilder:
    """Builds a PersonalProfile from datalake content."""

    def __init__(self, datalake_path: Path, graph_stats: dict | None = None):
        self.analyzer = DatalakeAnalyzer(datalake_path)
        self.graph_stats = graph_stats or {}

    def build(self, output_path: Path | None = None) -> PersonalProfile:
        """Build the profile from all available data sources."""
        tp_data = self.analyzer.analyze_training_pairs()
        ac_data = self.analyzer.analyze_auto_captures()

        total = tp_data["total_pairs"] + ac_data["total_captures"]

        # Detect domain
        domain, confidence = self._detect_domain(tp_data, ac_data)

        # Compute topic weights from categories
        top_topics = self._compute_topic_weights(tp_data["categories"])

        # Detect task types from categories
        task_types = self._detect_task_types(tp_data["categories"])

        # Detect style
        style = self._detect_style(tp_data)

        # Build patterns from file extensions and categories
        patterns = self._detect_patterns(ac_data, tp_data)

        profile = PersonalProfile(
            domain=domain,
            domain_confidence=confidence,
            top_topics=top_topics,
            style=style,
            patterns=patterns,
            task_types_detected=task_types,
            total_entries=total,
            built_at=datetime.now().isoformat(),
        )

        if output_path:
            save_profile(profile, output_path)

        logger.info(
            "profile_built",
            domain=domain,
            confidence=round(confidence, 2),
            topics=len(top_topics),
            entries=total,
        )

        return profile

    def _detect_domain(self, tp_data: dict, ac_data: dict) -> tuple[str, float]:
        """Detect the user's primary domain from data signals."""
        if tp_data["total_pairs"] == 0 and ac_data["total_captures"] == 0:
            return "unknown", 0.0

        code_signals = 0
        total_signals = 0

        # Check categories for code-related ones
        for cat, count in tp_data["categories"].items():
            total_signals += count
            if cat.lower() in CODE_CATEGORIES:
                code_signals += count

        # Check file extensions
        for ext, count in ac_data.get("file_extensions", {}).items():
            total_signals += count
            if ext in CODE_EXTENSIONS:
                code_signals += count

        if total_signals == 0:
            return "unknown", 0.0

        code_ratio = code_signals / total_signals

        if code_ratio > 0.5:
            return "software_development", min(code_ratio + 0.1, 1.0)

        # Not code-dominant: derive domain from top category
        if tp_data["categories"]:
            top_cat = max(tp_data["categories"], key=tp_data["categories"].get)
            return top_cat, min(tp_data["categories"][top_cat] / total_signals + 0.2, 1.0)

        return "unknown", 0.0

    def _compute_topic_weights(self, categories: dict) -> list[TopicWeight]:
        """Compute normalized topic weights from category counts."""
        if not categories:
            return []

        total = sum(categories.values())
        topics = [
            TopicWeight(topic=cat, weight=round(count / total, 3))
            for cat, count in sorted(categories.items(), key=lambda x: -x[1])
        ]
        return topics[:10]  # Top 10

    def _detect_task_types(self, categories: dict) -> list[str]:
        """Detect task types from categories."""
        task_type_map = {
            "code-review": "review",
            "debugging": "problem_solving",
            "testing": "review",
            "refactoring": "analysis",
            "agents": "creation",
            "ddd": "analysis",
        }
        detected = set()
        for cat in categories:
            if mapped := task_type_map.get(cat.lower()):
                detected.add(mapped)
            else:
                detected.add(cat.lower())
        return sorted(detected) if detected else ["general"]

    def _detect_style(self, tp_data: dict) -> StyleProfile:
        """Detect communication style from training pairs."""
        # Simple heuristic: detect language from categories/tags
        es_signals = sum(1 for t in tp_data.get("tags", {}) if any(
            w in t.lower() for w in ["español", "consulta", "derecho", "ley"]
        ))
        lang = "es" if es_signals > 2 else "en"
        return StyleProfile(formality=0.5, verbosity=0.5, language=lang)

    def _detect_patterns(self, ac_data: dict, tp_data: dict) -> list[str]:
        """Detect behavioral patterns from data."""
        patterns = []

        # File extension patterns
        exts = ac_data.get("file_extensions", {})
        if exts:
            top_ext = max(exts, key=exts.get)
            ext_map = {".py": "Python", ".ts": "TypeScript", ".tsx": "React/TypeScript",
                       ".js": "JavaScript", ".java": "Java", ".go": "Go", ".rs": "Rust"}
            if lang := ext_map.get(top_ext):
                patterns.append(f"Primary language: {lang}")

        # Project count
        projects = ac_data.get("projects", {})
        if len(projects) > 3:
            patterns.append(f"Works across {len(projects)} projects")
        elif len(projects) == 1:
            top_proj = list(projects.keys())[0]
            patterns.append(f"Focused on project: {top_proj}")

        # Top category patterns
        cats = tp_data.get("categories", {})
        if cats:
            top3 = sorted(cats, key=cats.get, reverse=True)[:3]
            patterns.append(f"Top expertise: {', '.join(top3)}")

        return patterns
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_personal_profile.py::TestProfileBuilder -v --tb=short`
Expected: 5 passed

**Step 5: Commit**

```
FEAT: Add ProfileBuilder with domain detection
```

---

### Task 4: CLI Commands — `fabrik profile`

**Files:**
- Modify: `src/interfaces/cli.py`
- Test: `tests/test_cli.py` (add profile tests)

**Step 1: Write the failing test — CLI profile command**

Add to `tests/test_cli.py`:

```python
class TestProfileCommand:
    """Test fabrik profile CLI command."""

    def test_profile_show_no_profile(self):
        from typer.testing import CliRunner
        from src.interfaces.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["profile", "show"])
        assert result.exit_code == 0
        assert "No profile" in result.output or "unknown" in result.output.lower()

    def test_profile_build(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner
        from src.interfaces.cli import app
        from src.config import settings

        # Point to tmp datalake with sample data
        tp_dir = tmp_path / "02-processed" / "training-pairs"
        tp_dir.mkdir(parents=True)
        pairs = [
            {"instruction": "test", "output": "ok", "category": "testing", "tags": ["testing"]}
        ]
        (tp_dir / "test.jsonl").write_text("\n".join(json.dumps(p) for p in pairs))

        monkeypatch.setattr(settings, "datalake_path", tmp_path)
        monkeypatch.setattr(settings, "data_dir", tmp_path / "data")

        runner = CliRunner()
        result = runner.invoke(app, ["profile", "build"])
        assert result.exit_code == 0
        assert "built" in result.output.lower() or "profile" in result.output.lower()
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_cli.py::TestProfileCommand -v --tb=short`
Expected: FAIL — no "profile" command

**Step 3: Write minimal implementation**

Add to `src/interfaces/cli.py`, after the existing commands:

```python
@app.command()
def profile(
    action: str = typer.Argument("show", help="Action: show, build"),
):
    """Manage your personal profile — learned from the datalake."""
    from src.core.personal_profile import ProfileBuilder, load_profile, save_profile

    profile_path = settings.data_dir / "profile" / "personal_profile.json"

    if action == "build":
        console.print(Panel.fit("[bold]Building Personal Profile[/bold]"))

        builder = ProfileBuilder(datalake_path=settings.datalake_path)
        result = builder.build(output_path=profile_path)

        table = Table(title="Profile Built")
        table.add_column("Field", style="bold")
        table.add_column("Value")
        table.add_row("Domain", f"{result.domain} ({result.domain_confidence:.0%})")
        table.add_row("Topics", ", ".join(t.topic for t in result.top_topics[:5]))
        table.add_row("Patterns", str(len(result.patterns)))
        table.add_row("Task Types", ", ".join(result.task_types_detected))
        table.add_row("Total Entries", str(result.total_entries))
        table.add_row("Saved To", str(profile_path))
        console.print(table)

    elif action == "show":
        loaded = load_profile(profile_path)
        if loaded.domain == "unknown" and not loaded.top_topics:
            console.print("[yellow]No profile built yet. Run:[/yellow] fabrik profile build")
            return

        table = Table(title=f"Personal Profile — {loaded.domain.replace('_', ' ').title()}")
        table.add_column("Field", style="bold")
        table.add_column("Value")
        table.add_row("Domain", f"{loaded.domain} ({loaded.domain_confidence:.0%})")
        table.add_row("Built", loaded.built_at or "unknown")
        table.add_row("Total Entries", str(loaded.total_entries))

        if loaded.top_topics:
            topics_str = "\n".join(
                f"  {t.topic}: {t.weight:.0%}" for t in loaded.top_topics[:8]
            )
            table.add_row("Top Topics", topics_str)

        if loaded.patterns:
            table.add_row("Patterns", "\n".join(f"  {p}" for p in loaded.patterns))

        if loaded.task_types_detected:
            table.add_row("Task Types", ", ".join(loaded.task_types_detected))

        table.add_row("Style", f"Formality: {loaded.style.formality:.0%}, Language: {loaded.style.language}")
        console.print(table)

        console.print(f"\n[dim]System prompt preview:[/dim]")
        console.print(f"[italic]{loaded.to_system_prompt()}[/italic]")
    else:
        console.print("[yellow]Usage:[/yellow] fabrik profile [show|build]")
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_cli.py::TestProfileCommand -v --tb=short`
Expected: 2 passed

**Step 5: Commit**

```
FEAT: Add fabrik profile CLI command
```

---

### Task 5: LLM Integration — Inject profile into system prompt

**Files:**
- Modify: `src/interfaces/cli.py` (ask and chat commands)
- Test: `tests/test_personal_profile.py`

**Step 1: Write the failing test — Profile injection**

Add to `tests/test_personal_profile.py`:

```python
class TestProfileIntegration:
    """Test profile integration with LLM calls."""

    def test_get_active_profile_returns_loaded(self, tmp_path):
        from src.core.personal_profile import (
            PersonalProfile, TopicWeight, save_profile, get_active_profile
        )

        profile_path = tmp_path / "profile.json"
        save_profile(
            PersonalProfile(domain="testing", top_topics=[TopicWeight(topic="pytest", weight=1.0)]),
            profile_path,
        )
        active = get_active_profile(profile_path)
        assert active.domain == "testing"

    def test_get_active_profile_caches(self, tmp_path):
        from src.core.personal_profile import (
            PersonalProfile, save_profile, get_active_profile, _profile_cache
        )

        _profile_cache.clear()
        profile_path = tmp_path / "profile.json"
        save_profile(PersonalProfile(domain="cached"), profile_path)

        p1 = get_active_profile(profile_path)
        p2 = get_active_profile(profile_path)
        assert p1 is p2  # Same object = cached

    def test_get_active_profile_missing_returns_empty(self, tmp_path):
        from src.core.personal_profile import get_active_profile, _profile_cache

        _profile_cache.clear()
        active = get_active_profile(tmp_path / "nope.json")
        assert active.domain == "unknown"
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_personal_profile.py::TestProfileIntegration -v --tb=short`
Expected: FAIL — `ImportError: cannot import name 'get_active_profile'`

**Step 3: Write minimal implementation**

Add to `src/core/personal_profile.py`:

```python
# Simple cache to avoid re-reading profile on every LLM call
_profile_cache: dict[str, PersonalProfile] = {}


def get_active_profile(profile_path: Path | None = None) -> PersonalProfile:
    """Get the active profile, with simple caching."""
    from src.config import settings

    path = profile_path or settings.data_dir / "profile" / "personal_profile.json"
    cache_key = str(path)

    if cache_key in _profile_cache:
        return _profile_cache[cache_key]

    profile = load_profile(path)
    _profile_cache[cache_key] = profile
    return profile
```

Then modify the `ask` command in `src/interfaces/cli.py` to inject the profile.
Find the section where `client.generate()` is called and add the profile system prompt:

```python
# In the ask command's run() async function, before client.generate():
from src.core.personal_profile import get_active_profile

profile = get_active_profile()
system_prompt = profile.to_system_prompt()

# Pass system prompt to generate call
response = await client.generate(final_prompt, system=system_prompt)
```

**Step 4: Run tests to verify all pass**

Run: `python3 -m pytest tests/test_personal_profile.py -v --tb=short`
Expected: All passed (schema + analyzer + builder + integration)

Run: `python3 -m pytest tests/ -q --tb=short`
Expected: All 527+ tests pass

**Step 5: Commit**

```
FEAT: Inject profile into LLM system prompt
```

---

### Task 6: Full integration test with real datalake

**Files:**
- Test: `tests/test_personal_profile.py`

**Step 1: Write integration test**

Add to `tests/test_personal_profile.py`:

```python
class TestRealDatalakeIntegration:
    """Integration test with the actual datalake (skipped in CI)."""

    @pytest.mark.skipif(
        not Path("/path/to/datalake").exists(),
        reason="Real datalake not available",
    )
    def test_build_from_real_datalake(self):
        from src.core.personal_profile import ProfileBuilder

        builder = ProfileBuilder(
            datalake_path=Path("/path/to/datalake")
        )
        profile = builder.build()

        # Should detect software development
        assert profile.domain == "software_development"
        assert profile.domain_confidence > 0.7
        assert profile.total_entries > 1000
        assert len(profile.top_topics) >= 5
        assert any(t.topic == "postgresql" for t in profile.top_topics)
        assert len(profile.patterns) > 0
```

**Step 2: Run it**

Run: `python3 -m pytest tests/test_personal_profile.py::TestRealDatalakeIntegration -v`
Expected: PASS (on local machine with datalake)

**Step 3: Run full test suite**

Run: `python3 -m pytest tests/ -q --tb=short`
Expected: 540+ tests pass (527 existing + 15+ new)

**Step 4: Final commit**

```
FEAT: Complete Personal Profile with integration test
```

---

## Summary

| Task | What | Tests | Files |
|------|------|-------|-------|
| 1 | Profile data model + serialization | 6 | `personal_profile.py`, `test_personal_profile.py` |
| 2 | Datalake analyzer (training pairs + captures) | 3 | `personal_profile.py` |
| 3 | Domain detection + ProfileBuilder | 5 | `personal_profile.py` |
| 4 | CLI `fabrik profile show/build` | 2 | `cli.py`, `test_cli.py` |
| 5 | LLM integration (system prompt injection) | 3 | `personal_profile.py`, `cli.py` |
| 6 | Real datalake integration test | 1 | `test_personal_profile.py` |

**Total: 6 tasks, ~20 tests, 2 new files, 1 modified file**
