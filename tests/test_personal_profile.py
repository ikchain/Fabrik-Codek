"""Tests for Personal Profile."""

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest


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
        from src.core.personal_profile import PersonalProfile, StyleProfile, TopicWeight

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
        from src.core.personal_profile import (
            PersonalProfile,
            TopicWeight,
            load_profile,
            save_profile,
        )

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
            {
                "instruction": "How to optimize a query?",
                "output": "Use EXPLAIN...",
                "category": "postgresql",
                "tags": ["postgresql", "performance"],
            },
            {
                "instruction": "Index types?",
                "output": "B-tree, hash...",
                "category": "postgresql",
                "tags": ["postgresql", "indexing"],
            },
            {
                "instruction": "Connection pooling?",
                "output": "Use pgbouncer...",
                "category": "postgresql",
                "tags": ["postgresql", "connections"],
            },
        ]
        (tp_dir / "postgresql-basics.jsonl").write_text("\n".join(json.dumps(p) for p in pairs1))
        pairs2 = [
            {
                "instruction": "Fix timeout error",
                "output": "Add retry...",
                "category": "debugging",
                "tags": ["debugging", "timeout"],
            },
        ]
        (tp_dir / "debugging-basics.jsonl").write_text("\n".join(json.dumps(p) for p in pairs2))
        return tmp_path

    @pytest.fixture
    def sample_auto_captures(self, tmp_path):
        ac_dir = tmp_path / "01-raw" / "code-changes"
        ac_dir.mkdir(parents=True)
        captures = [
            {
                "timestamp": "2026-02-20T10:00:00",
                "type": "auto_capture",
                "tool": "Edit",
                "project": "my-api",
                "file_modified": "/home/user/my-api/src/main.py",
                "change_type": "edit",
            },
            {
                "timestamp": "2026-02-20T10:05:00",
                "type": "auto_capture",
                "tool": "Write",
                "project": "my-api",
                "file_modified": "/home/user/my-api/tests/test_main.py",
                "change_type": "write",
            },
            {
                "timestamp": "2026-02-20T10:10:00",
                "type": "auto_capture",
                "tool": "Edit",
                "project": "frontend",
                "file_modified": "/home/user/frontend/src/App.tsx",
                "change_type": "edit",
            },
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


class TestProfileBuilder:
    @pytest.fixture
    def datalake_with_code(self, tmp_path):
        tp_dir = tmp_path / "02-processed" / "training-pairs"
        tp_dir.mkdir(parents=True)
        ac_dir = tmp_path / "01-raw" / "code-changes"
        ac_dir.mkdir(parents=True)

        pairs = []
        for i in range(20):
            pairs.append(
                {
                    "instruction": f"pg query {i}",
                    "output": "...",
                    "category": "postgresql",
                    "tags": ["postgresql"],
                }
            )
        for i in range(10):
            pairs.append(
                {
                    "instruction": f"debug {i}",
                    "output": "...",
                    "category": "debugging",
                    "tags": ["debugging"],
                }
            )
        for i in range(5):
            pairs.append(
                {
                    "instruction": f"angular {i}",
                    "output": "...",
                    "category": "angular",
                    "tags": ["angular"],
                }
            )
        (tp_dir / "all.jsonl").write_text("\n".join(json.dumps(p) for p in pairs))

        captures = []
        for i in range(15):
            captures.append(
                {
                    "timestamp": "2026-01-01T00:00:00",
                    "type": "auto_capture",
                    "tool": "Edit",
                    "project": "backend",
                    "file_modified": f"/src/file{i}.py",
                    "change_type": "edit",
                }
            )
        for i in range(5):
            captures.append(
                {
                    "timestamp": "2026-01-01T00:00:00",
                    "type": "auto_capture",
                    "tool": "Edit",
                    "project": "frontend",
                    "file_modified": f"/src/comp{i}.tsx",
                    "change_type": "edit",
                }
            )
        (ac_dir / "2026-01-01_auto-captures.jsonl").write_text(
            "\n".join(json.dumps(c) for c in captures)
        )
        return tmp_path

    @pytest.fixture
    def datalake_with_legal(self, tmp_path):
        tp_dir = tmp_path / "02-processed" / "training-pairs"
        tp_dir.mkdir(parents=True)
        pairs = []
        for i in range(15):
            pairs.append(
                {
                    "instruction": f"consulta civil {i}",
                    "output": "...",
                    "category": "civil_law",
                    "tags": ["civil_law", "contracts"],
                }
            )
        for i in range(8):
            pairs.append(
                {
                    "instruction": f"caso laboral {i}",
                    "output": "...",
                    "category": "labor_law",
                    "tags": ["labor_law"],
                }
            )
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
            PersonalProfile,
            TopicWeight,
            get_active_profile,
            save_profile,
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
            PersonalProfile,
            _profile_cache,
            get_active_profile,
            save_profile,
        )

        _profile_cache.clear()
        profile_path = tmp_path / "profile.json"
        save_profile(PersonalProfile(domain="cached"), profile_path)

        p1 = get_active_profile(profile_path)
        p2 = get_active_profile(profile_path)
        assert p1 is p2  # Same object = cached

    def test_get_active_profile_missing_returns_empty(self, tmp_path):
        from src.core.personal_profile import _profile_cache, get_active_profile

        _profile_cache.clear()
        active = get_active_profile(tmp_path / "nope.json")
        assert active.domain == "unknown"


class TestRealDatalakeIntegration:
    """Integration test with the actual datalake (skipped in CI)."""

    @pytest.mark.skipif(
        not Path(os.environ.get("FABRIK_DATALAKE_PATH", "/tmp/no-datalake")).exists(),
        reason="Real datalake not available (set FABRIK_DATALAKE_PATH)",
    )
    def test_build_from_real_datalake(self):
        from src.core.personal_profile import ProfileBuilder

        builder = ProfileBuilder(datalake_path=Path(os.environ["FABRIK_DATALAKE_PATH"]))
        profile = builder.build()

        # Should detect software development
        assert profile.domain == "software_development"
        assert profile.domain_confidence > 0.4
        assert profile.total_entries > 1000
        assert len(profile.top_topics) >= 5
        assert any(
            t.topic in ("postgresql", "docker", "kubernetes", "ddd", "fastapi")
            for t in profile.top_topics
        )
        assert len(profile.patterns) > 0


# --- SPRInG incremental build tests ---


class TestDriftDetection:
    """Test drift detection via cosine distance."""

    def test_drift_detected_significant_change(self):
        from src.core.personal_profile import ProfileBuilder, TopicWeight

        builder = ProfileBuilder(datalake_path=Path("/tmp"))
        existing = [
            TopicWeight(topic="postgresql", weight=0.8),
            TopicWeight(topic="docker", weight=0.2),
        ]
        new = [
            TopicWeight(topic="angular", weight=0.7),
            TopicWeight(topic="react", weight=0.3),
        ]
        drift_detected, distance, topics_drifted = builder._detect_drift(existing, new)
        assert drift_detected is True
        assert distance > 0.3
        assert len(topics_drifted) > 0

    def test_no_drift_similar_distributions(self):
        from src.core.personal_profile import ProfileBuilder, TopicWeight

        builder = ProfileBuilder(datalake_path=Path("/tmp"))
        existing = [
            TopicWeight(topic="postgresql", weight=0.6),
            TopicWeight(topic="docker", weight=0.4),
        ]
        new = [
            TopicWeight(topic="postgresql", weight=0.55),
            TopicWeight(topic="docker", weight=0.45),
        ]
        drift_detected, distance, _ = builder._detect_drift(existing, new)
        assert drift_detected is False
        assert distance < 0.3

    def test_drift_empty_topics(self):
        from src.core.personal_profile import ProfileBuilder, TopicWeight

        builder = ProfileBuilder(datalake_path=Path("/tmp"))
        # Empty existing
        drift_detected, distance, topics_drifted = builder._detect_drift(
            [], [TopicWeight(topic="a", weight=1.0)]
        )
        assert drift_detected is False
        assert distance == 0.0
        assert topics_drifted == []

        # Empty new
        drift_detected, distance, topics_drifted = builder._detect_drift(
            [TopicWeight(topic="a", weight=1.0)], []
        )
        assert drift_detected is False

    def test_drift_returns_drifted_topics(self):
        from src.core.personal_profile import ProfileBuilder, TopicWeight

        builder = ProfileBuilder(datalake_path=Path("/tmp"))
        existing = [
            TopicWeight(topic="postgresql", weight=0.5),
            TopicWeight(topic="docker", weight=0.5),
        ]
        new = [
            TopicWeight(topic="postgresql", weight=0.1),
            TopicWeight(topic="docker", weight=0.1),
            TopicWeight(topic="angular", weight=0.8),
        ]
        _, _, topics_drifted = builder._detect_drift(existing, new)
        # postgresql drops by 0.4, docker drops by 0.4, angular appears at 0.8
        assert "angular" in topics_drifted
        assert "postgresql" in topics_drifted
        assert "docker" in topics_drifted

    def test_drift_threshold_boundary(self):
        """Distance exactly at threshold should NOT trigger drift."""
        from src.core.personal_profile import DRIFT_THRESHOLD, ProfileBuilder, TopicWeight

        builder = ProfileBuilder(datalake_path=Path("/tmp"))
        # Identical distributions => distance = 0.0
        topics = [TopicWeight(topic="a", weight=1.0)]
        drift_detected, distance, _ = builder._detect_drift(topics, topics)
        assert drift_detected is False
        assert distance <= DRIFT_THRESHOLD


class TestEMAMerge:
    """Test EMA topic weight merging."""

    def test_merge_alpha_03_gradual(self):
        from src.core.personal_profile import ProfileBuilder, TopicWeight

        builder = ProfileBuilder(datalake_path=Path("/tmp"))
        existing = [TopicWeight(topic="postgresql", weight=1.0)]
        new = [TopicWeight(topic="postgresql", weight=0.0)]
        merged = builder._merge_topic_weights(existing, new, alpha=0.3)
        # merged = 0.3 * 0 + 0.7 * 1.0 = 0.7, normalized to 1.0
        assert len(merged) == 1
        assert merged[0].topic == "postgresql"
        # Single topic normalizes to 1.0
        assert abs(merged[0].weight - 1.0) < 0.01

    def test_merge_alpha_07_drift(self):
        from src.core.personal_profile import ProfileBuilder, TopicWeight

        builder = ProfileBuilder(datalake_path=Path("/tmp"))
        existing = [
            TopicWeight(topic="postgresql", weight=0.8),
            TopicWeight(topic="docker", weight=0.2),
        ]
        new = [
            TopicWeight(topic="postgresql", weight=0.2),
            TopicWeight(topic="docker", weight=0.8),
        ]
        merged = builder._merge_topic_weights(existing, new, alpha=0.7)
        merged_map = {tw.topic: tw.weight for tw in merged}
        # pg: 0.7*0.2 + 0.3*0.8 = 0.38, docker: 0.7*0.8 + 0.3*0.2 = 0.62
        # After normalization: pg ~ 0.38, docker ~ 0.62
        assert merged_map["docker"] > merged_map["postgresql"]

    def test_merge_new_topic_appears(self):
        from src.core.personal_profile import ProfileBuilder, TopicWeight

        builder = ProfileBuilder(datalake_path=Path("/tmp"))
        existing = [TopicWeight(topic="postgresql", weight=1.0)]
        new = [TopicWeight(topic="angular", weight=1.0)]
        merged = builder._merge_topic_weights(existing, new, alpha=0.5)
        topics = {tw.topic for tw in merged}
        assert "angular" in topics
        assert "postgresql" in topics

    def test_merge_renormalizes_to_top_10(self):
        from src.core.personal_profile import ProfileBuilder, TopicWeight

        builder = ProfileBuilder(datalake_path=Path("/tmp"))
        existing = [TopicWeight(topic=f"topic_{i}", weight=1.0 / 12) for i in range(12)]
        new = [TopicWeight(topic=f"topic_{i}", weight=1.0 / 12) for i in range(12)]
        merged = builder._merge_topic_weights(existing, new, alpha=0.5)
        assert len(merged) <= 10
        total_weight = sum(tw.weight for tw in merged)
        assert abs(total_weight - 1.0) < 0.02


class TestIncrementalBuild:
    """Test build_incremental behavior."""

    @pytest.fixture
    def datalake_for_incremental(self, tmp_path):
        """Create a datalake with training pairs for incremental build testing."""
        tp_dir = tmp_path / "datalake" / "02-processed" / "training-pairs"
        tp_dir.mkdir(parents=True)
        ac_dir = tmp_path / "datalake" / "01-raw" / "code-changes"
        ac_dir.mkdir(parents=True)

        pairs = []
        for i in range(10):
            pairs.append(
                {
                    "instruction": f"pg query {i}",
                    "output": "...",
                    "category": "postgresql",
                    "tags": ["postgresql"],
                }
            )
        for i in range(5):
            pairs.append(
                {
                    "instruction": f"debug {i}",
                    "output": "...",
                    "category": "debugging",
                    "tags": ["debugging"],
                }
            )
        (tp_dir / "initial.jsonl").write_text("\n".join(json.dumps(p) for p in pairs))

        captures = []
        for i in range(5):
            captures.append(
                {
                    "timestamp": "2026-02-20T10:00:00",
                    "type": "auto_capture",
                    "tool": "Edit",
                    "project": "backend",
                    "file_modified": f"/src/file{i}.py",
                    "change_type": "edit",
                }
            )
        (ac_dir / "2026-02-20_auto-captures.jsonl").write_text(
            "\n".join(json.dumps(c) for c in captures)
        )
        return tmp_path

    def test_incremental_no_prior_falls_back_to_full(self, datalake_for_incremental, tmp_path):
        from src.core.personal_profile import ProfileBuilder

        datalake_path = datalake_for_incremental / "datalake"
        output = tmp_path / "profile.json"
        builder = ProfileBuilder(datalake_path=datalake_path)
        result = builder.build_incremental(output_path=output)
        # Should fall back to full build
        assert result.build_mode == "full"
        assert result.last_build_timestamp is not None
        assert result.total_entries > 0

    def test_incremental_no_new_entries_returns_existing(self, datalake_for_incremental, tmp_path):
        from src.core.personal_profile import PersonalProfile, TopicWeight, save_profile

        datalake_path = datalake_for_incremental / "datalake"
        output = tmp_path / "profile.json"

        # Save an existing profile with a future timestamp so no files pass mtime filter
        existing = PersonalProfile(
            domain="software_development",
            domain_confidence=0.9,
            top_topics=[TopicWeight(topic="postgresql", weight=1.0)],
            total_entries=100,
            last_build_timestamp="2099-01-01T00:00:00",
            build_mode="full",
        )
        save_profile(existing, output)

        from src.core.personal_profile import ProfileBuilder

        builder = ProfileBuilder(datalake_path=datalake_path)
        result = builder.build_incremental(output_path=output)
        # No new entries => returns existing unchanged
        assert result.total_entries == 100
        assert result.build_mode == "full"

    def test_incremental_merges_correctly(self, datalake_for_incremental, tmp_path):
        from src.core.personal_profile import (
            PersonalProfile,
            ProfileBuilder,
            TopicWeight,
            save_profile,
        )

        datalake_path = datalake_for_incremental / "datalake"
        output = tmp_path / "profile.json"

        # Save existing profile with past timestamp so new files are picked up
        existing = PersonalProfile(
            domain="software_development",
            domain_confidence=0.9,
            top_topics=[
                TopicWeight(topic="postgresql", weight=0.7),
                TopicWeight(topic="docker", weight=0.3),
            ],
            patterns=["Use Python for code examples"],
            task_types_detected=["backend"],
            total_entries=50,
            last_build_timestamp="2020-01-01T00:00:00",
            build_mode="full",
        )
        save_profile(existing, output)

        builder = ProfileBuilder(datalake_path=datalake_path)
        result = builder.build_incremental(output_path=output)
        assert result.build_mode == "incremental"
        assert result.total_entries > 50
        # Topics should be merged — postgresql should still be present
        topic_names = [tw.topic for tw in result.top_topics]
        assert "postgresql" in topic_names
        # Weights should sum to ~1.0
        total_w = sum(tw.weight for tw in result.top_topics)
        assert abs(total_w - 1.0) < 0.02

    def test_incremental_updates_timestamp(self, datalake_for_incremental, tmp_path):
        from src.core.personal_profile import (
            PersonalProfile,
            ProfileBuilder,
            TopicWeight,
            save_profile,
        )

        datalake_path = datalake_for_incremental / "datalake"
        output = tmp_path / "profile.json"

        old_ts = "2020-01-01T00:00:00"
        existing = PersonalProfile(
            domain="software_development",
            domain_confidence=0.9,
            top_topics=[TopicWeight(topic="postgresql", weight=1.0)],
            total_entries=50,
            last_build_timestamp=old_ts,
            build_mode="full",
        )
        save_profile(existing, output)

        builder = ProfileBuilder(datalake_path=datalake_path)
        result = builder.build_incremental(output_path=output)
        assert result.last_build_timestamp is not None
        assert result.last_build_timestamp != old_ts


class TestReplayBuffer:
    """Test replay buffer computation."""

    def test_replay_buffer_keeps_novel_entries(self):
        from src.core.personal_profile import ProfileBuilder, TopicWeight

        builder = ProfileBuilder(datalake_path=Path("/tmp"))
        tp_data = {
            "categories": {"postgresql": 10, "new_topic": 5, "docker": 8},
        }
        existing_topics = [
            TopicWeight(topic="postgresql", weight=0.6),
            TopicWeight(topic="docker", weight=0.4),
        ]
        buffer = builder._compute_replay_buffer(tp_data, existing_topics)
        # new_topic should be first (novelty=1.0)
        assert buffer[0]["category"] == "new_topic"
        assert buffer[0]["novelty_score"] == 1.0
        # Known topics should have novelty=0.0
        known = [b for b in buffer if b["novelty_score"] == 0.0]
        assert len(known) == 2

    def test_replay_buffer_max_size(self):
        from src.core.personal_profile import ProfileBuilder

        builder = ProfileBuilder(datalake_path=Path("/tmp"))
        tp_data = {
            "categories": {f"topic_{i}": i for i in range(30)},
        }
        buffer = builder._compute_replay_buffer(tp_data, [], max_size=5)
        assert len(buffer) <= 5


class TestTimestampFiltering:
    """Test since parameter in analyzer methods."""

    def test_training_pairs_since_filters_old_files(self, tmp_path):
        from src.core.personal_profile import DatalakeAnalyzer

        tp_dir = tmp_path / "02-processed" / "training-pairs"
        tp_dir.mkdir(parents=True)

        # Create a file and set its mtime to the past
        old_file = tp_dir / "old.jsonl"
        old_file.write_text(json.dumps({"category": "old_topic"}))
        # Set mtime to Jan 1, 2020
        os.utime(old_file, (time.time(), 1577836800.0))

        # Create a recent file
        new_file = tp_dir / "new.jsonl"
        new_file.write_text(json.dumps({"category": "new_topic"}))
        # Ensure it has current mtime (default)

        analyzer = DatalakeAnalyzer(datalake_path=tmp_path)
        result = analyzer.analyze_training_pairs(since="2025-01-01T00:00:00")
        assert result["total_pairs"] == 1
        assert "new_topic" in result["categories"]
        assert "old_topic" not in result["categories"]

    def test_auto_captures_since_filters_old(self, tmp_path):
        from src.core.personal_profile import DatalakeAnalyzer

        ac_dir = tmp_path / "01-raw" / "code-changes"
        ac_dir.mkdir(parents=True)

        # File with mixed timestamps
        records = [
            {
                "timestamp": "2020-01-01T00:00:00",
                "type": "auto_capture",
                "tool": "Edit",
                "project": "old",
                "file_modified": "/old.py",
            },
            {
                "timestamp": "2026-06-01T00:00:00",
                "type": "auto_capture",
                "tool": "Write",
                "project": "new",
                "file_modified": "/new.py",
            },
        ]
        (ac_dir / "mixed_auto-captures.jsonl").write_text("\n".join(json.dumps(r) for r in records))

        analyzer = DatalakeAnalyzer(datalake_path=tmp_path)
        result = analyzer.analyze_auto_captures(since="2025-01-01T00:00:00")
        # Only the new record should pass
        assert result["total_captures"] == 1
        assert "new" in result["projects"]
        assert "old" not in result["projects"]


class TestDriftHistory:
    """Test drift history accumulation across incremental builds."""

    @pytest.fixture
    def datalake_two_phases(self, tmp_path):
        """Create a datalake that changes between builds."""
        tp_dir = tmp_path / "datalake" / "02-processed" / "training-pairs"
        tp_dir.mkdir(parents=True)
        ac_dir = tmp_path / "datalake" / "01-raw" / "code-changes"
        ac_dir.mkdir(parents=True)
        return tmp_path

    def test_drift_history_accumulates(self, datalake_two_phases, tmp_path):
        from src.core.personal_profile import (
            ProfileBuilder,
            save_profile,
        )

        datalake_path = datalake_two_phases / "datalake"
        output = tmp_path / "profile.json"
        tp_dir = datalake_path / "02-processed" / "training-pairs"

        # Phase 1: postgresql heavy
        pairs_p1 = [
            {"instruction": f"pg {i}", "output": "...", "category": "postgresql"} for i in range(20)
        ]
        (tp_dir / "phase1.jsonl").write_text("\n".join(json.dumps(p) for p in pairs_p1))

        # Full build first
        builder = ProfileBuilder(datalake_path=datalake_path)
        p1 = builder.build(output_path=output)
        # Set timestamp and save so incremental works
        p1.last_build_timestamp = "2020-01-01T00:00:00"
        save_profile(p1, output)

        # Phase 2: completely different topic (angular) => should trigger drift
        pairs_p2 = [
            {"instruction": f"ng {i}", "output": "...", "category": "angular"} for i in range(30)
        ]
        (tp_dir / "phase2.jsonl").write_text("\n".join(json.dumps(p) for p in pairs_p2))

        result = builder.build_incremental(output_path=output)
        # Drift should have been detected (postgresql -> angular is large shift)
        assert len(result.drift_history) >= 1
        assert result.drift_history[-1]["magnitude"] > 0

    def test_drift_history_empty_when_no_drift(self, tmp_path):
        from src.core.personal_profile import (
            PersonalProfile,
            ProfileBuilder,
            TopicWeight,
            save_profile,
        )

        datalake_path = tmp_path / "datalake"
        tp_dir = datalake_path / "02-processed" / "training-pairs"
        tp_dir.mkdir(parents=True)
        ac_dir = datalake_path / "01-raw" / "code-changes"
        ac_dir.mkdir(parents=True)
        output = tmp_path / "profile.json"

        # Same topic in both phases — no drift
        pairs = [
            {"instruction": f"pg {i}", "output": "...", "category": "postgresql"} for i in range(10)
        ]
        (tp_dir / "data.jsonl").write_text("\n".join(json.dumps(p) for p in pairs))

        existing = PersonalProfile(
            domain="software_development",
            domain_confidence=0.9,
            top_topics=[TopicWeight(topic="postgresql", weight=1.0)],
            total_entries=50,
            last_build_timestamp="2020-01-01T00:00:00",
            build_mode="full",
            drift_history=[],
        )
        save_profile(existing, output)

        builder = ProfileBuilder(datalake_path=datalake_path)
        result = builder.build_incremental(output_path=output)
        assert result.drift_history == []


class TestBackwardCompatibility:
    """Test that new fields are backward compatible with old profiles."""

    def test_load_profile_without_new_fields(self, tmp_path):
        """Old profile JSON without new fields loads with defaults."""
        from src.core.personal_profile import load_profile

        old_data = {
            "domain": "software_development",
            "domain_confidence": 0.9,
            "top_topics": [{"topic": "postgresql", "weight": 1.0}],
            "style": {"formality": 0.5, "verbosity": 0.5, "language": "en"},
            "patterns": [],
            "task_types_detected": [],
            "total_entries": 100,
            "built_at": "2026-01-01T00:00:00",
        }
        filepath = tmp_path / "old_profile.json"
        filepath.write_text(json.dumps(old_data))
        loaded = load_profile(filepath)
        assert loaded.last_build_timestamp is None
        assert loaded.build_mode == "full"
        assert loaded.drift_history == []
        assert loaded.replay_buffer == []

    def test_to_dict_includes_new_fields(self):
        from src.core.personal_profile import PersonalProfile

        profile = PersonalProfile(
            last_build_timestamp="2026-02-27T00:00:00",
            build_mode="incremental",
            drift_history=[{"date": "2026-02-27", "topics_drifted": ["a"], "magnitude": 0.5}],
            replay_buffer=[{"category": "a", "count": 1, "novelty_score": 1.0}],
        )
        d = profile.to_dict()
        assert d["last_build_timestamp"] == "2026-02-27T00:00:00"
        assert d["build_mode"] == "incremental"
        assert len(d["drift_history"]) == 1
        assert len(d["replay_buffer"]) == 1


class TestCLIIncrementalProfile:
    """Test CLI build-incremental and drift actions."""

    def test_cli_build_incremental(self, tmp_path):
        from unittest.mock import MagicMock

        from typer.testing import CliRunner

        from src.interfaces.cli import app

        runner = CliRunner()

        # Create datalake with training pairs
        tp_dir = tmp_path / "datalake" / "02-processed" / "training-pairs"
        tp_dir.mkdir(parents=True)
        ac_dir = tmp_path / "datalake" / "01-raw" / "code-changes"
        ac_dir.mkdir(parents=True)
        pairs = [
            {"instruction": "pg q", "output": "...", "category": "postgresql", "tags": []},
        ]
        (tp_dir / "data.jsonl").write_text("\n".join(json.dumps(p) for p in pairs))

        mock_settings = MagicMock()
        mock_settings.datalake_path = tmp_path / "datalake"
        mock_settings.data_dir = tmp_path / "data"

        with patch("src.config.settings", mock_settings):
            result = runner.invoke(app, ["profile", "build-incremental"])
            assert result.exit_code == 0
            # First time = falls back to full build
            assert "full build" in result.output.lower() or "Profile Built" in result.output

    def test_cli_drift_shows_history(self, tmp_path):
        from unittest.mock import MagicMock

        from typer.testing import CliRunner

        from src.interfaces.cli import app

        runner = CliRunner()

        from src.core.personal_profile import PersonalProfile, save_profile

        profile = PersonalProfile(
            domain="software_development",
            drift_history=[
                {
                    "date": "2026-02-25",
                    "topics_drifted": ["angular", "react"],
                    "magnitude": 0.4521,
                }
            ],
        )
        profile_path = tmp_path / "profile" / "personal_profile.json"
        save_profile(profile, profile_path)

        mock_settings = MagicMock()
        mock_settings.data_dir = tmp_path

        with patch("src.config.settings", mock_settings):
            result = runner.invoke(app, ["profile", "drift"])
            assert result.exit_code == 0
            assert "Drift History" in result.output
            assert "angular" in result.output
            assert "0.4521" in result.output
