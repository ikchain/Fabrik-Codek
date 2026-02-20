"""Tests for Personal Profile."""

import json
import pytest
import tempfile
from pathlib import Path


class TestProfileSchema:
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
            patterns=["Use Python for code examples", "Prefer FastAPI with async/await"],
            task_types_detected=["debugging", "code_review"],
        )
        prompt = profile.to_system_prompt()
        assert "software development" in prompt
        assert "FastAPI" in prompt
        assert "Python" in prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_empty_profile_gives_generic_prompt(self):
        from src.core.personal_profile import PersonalProfile
        profile = PersonalProfile()
        prompt = profile.to_system_prompt()
        assert "general" in prompt.lower() or len(prompt) < 200


class TestDatalakeAnalyzer:
    @pytest.fixture
    def sample_training_pairs(self, tmp_path):
        tp_dir = tmp_path / "02-processed" / "training-pairs"
        tp_dir.mkdir(parents=True)
        pairs1 = [
            {"instruction": "How to optimize a query?", "output": "Use EXPLAIN...", "category": "postgresql", "tags": ["postgresql", "performance"]},
            {"instruction": "Index types?", "output": "B-tree, hash...", "category": "postgresql", "tags": ["postgresql", "indexing"]},
            {"instruction": "Connection pooling?", "output": "Use pgbouncer...", "category": "postgresql", "tags": ["postgresql", "connections"]},
        ]
        (tp_dir / "postgresql-basics.jsonl").write_text("\n".join(json.dumps(p) for p in pairs1))
        pairs2 = [
            {"instruction": "Fix timeout error", "output": "Add retry...", "category": "debugging", "tags": ["debugging", "timeout"]},
        ]
        (tp_dir / "debugging-basics.jsonl").write_text("\n".join(json.dumps(p) for p in pairs2))
        return tmp_path

    @pytest.fixture
    def sample_auto_captures(self, tmp_path):
        ac_dir = tmp_path / "01-raw" / "code-changes"
        ac_dir.mkdir(parents=True)
        captures = [
            {"timestamp": "2026-02-20T10:00:00", "type": "auto_capture", "tool": "Edit", "project": "my-api", "file_modified": "/home/user/my-api/src/main.py", "change_type": "edit"},
            {"timestamp": "2026-02-20T10:05:00", "type": "auto_capture", "tool": "Write", "project": "my-api", "file_modified": "/home/user/my-api/tests/test_main.py", "change_type": "write"},
            {"timestamp": "2026-02-20T10:10:00", "type": "auto_capture", "tool": "Edit", "project": "frontend", "file_modified": "/home/user/frontend/src/App.tsx", "change_type": "edit"},
        ]
        (ac_dir / "2026-02-20_auto-captures.jsonl").write_text("\n".join(json.dumps(c) for c in captures))
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


class TestProfileBuilder:
    @pytest.fixture
    def datalake_with_code(self, tmp_path):
        tp_dir = tmp_path / "02-processed" / "training-pairs"
        tp_dir.mkdir(parents=True)
        ac_dir = tmp_path / "01-raw" / "code-changes"
        ac_dir.mkdir(parents=True)

        pairs = []
        for i in range(20):
            pairs.append({"instruction": f"pg query {i}", "output": "...", "category": "postgresql", "tags": ["postgresql"]})
        for i in range(10):
            pairs.append({"instruction": f"debug {i}", "output": "...", "category": "debugging", "tags": ["debugging"]})
        for i in range(5):
            pairs.append({"instruction": f"angular {i}", "output": "...", "category": "angular", "tags": ["angular"]})
        (tp_dir / "all.jsonl").write_text("\n".join(json.dumps(p) for p in pairs))

        captures = []
        for i in range(15):
            captures.append({"timestamp": "2026-01-01T00:00:00", "type": "auto_capture", "tool": "Edit", "project": "backend", "file_modified": f"/src/file{i}.py", "change_type": "edit"})
        for i in range(5):
            captures.append({"timestamp": "2026-01-01T00:00:00", "type": "auto_capture", "tool": "Edit", "project": "frontend", "file_modified": f"/src/comp{i}.tsx", "change_type": "edit"})
        (ac_dir / "2026-01-01_auto-captures.jsonl").write_text("\n".join(json.dumps(c) for c in captures))
        return tmp_path

    @pytest.fixture
    def datalake_with_legal(self, tmp_path):
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
        output = tmp_path / "out" / "profile.json"
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
        assert 0.9 <= total_weight <= 1.1


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
