"""Tests for Context-Map determinista."""

import json

from src.core.context_map import VALID_FRAGMENTS, ContextMap, ContextMapEntry, ContextMapResult

# ---------------------------------------------------------------------------
# ContextMapEntry
# ---------------------------------------------------------------------------


class TestContextMapEntry:
    def test_matches_simple_pattern(self):
        entry = ContextMapEntry(patterns=["^hola\\b"], task_type="general")
        assert entry.matches("hola mundo")
        assert not entry.matches("decir hola")  # ^ requires start

    def test_matches_case_insensitive(self):
        entry = ContextMapEntry(patterns=["^hola\\b"], task_type="general")
        assert entry.matches("Hola mundo")
        assert entry.matches("HOLA")

    def test_matches_multiple_patterns(self):
        entry = ContextMapEntry(
            patterns=["^hola\\b", "^hey\\b", "^buenos d[ií]as"],
            task_type="general",
        )
        assert entry.matches("hola")
        assert entry.matches("hey there")
        assert entry.matches("buenos dias")
        assert entry.matches("buenos días")
        assert not entry.matches("adios")

    def test_invalid_pattern_skipped(self):
        entry = ContextMapEntry(patterns=["[invalid", "^hola"], task_type="general")
        assert len(entry._compiled) == 1  # only valid pattern compiled
        assert entry.matches("hola")

    def test_no_match_returns_false(self):
        entry = ContextMapEntry(patterns=["^xyz$"], task_type="general")
        assert not entry.matches("abc")

    def test_empty_query(self):
        entry = ContextMapEntry(patterns=[".*"], task_type="general")
        assert entry.matches("")  # .* matches empty string

    def test_default_values(self):
        entry = ContextMapEntry(patterns=["test"])
        assert entry.task_type == "general"
        assert entry.use_context is False
        assert entry.profile_fragments == ["identity"]


# ---------------------------------------------------------------------------
# ContextMap.match
# ---------------------------------------------------------------------------


class TestContextMapMatch:
    def test_hit_returns_result(self):
        entry = ContextMapEntry(
            patterns=["^hola"],
            task_type="general",
            use_context=False,
            profile_fragments=["identity"],
        )
        cmap = ContextMap(entries=[entry])
        result = cmap.match("hola mundo")
        assert result is not None
        assert isinstance(result, ContextMapResult)
        assert result.entry is entry
        assert result.matched_pattern == "^hola"

    def test_miss_returns_none(self):
        entry = ContextMapEntry(patterns=["^hola"], task_type="general")
        cmap = ContextMap(entries=[entry])
        assert cmap.match("como funciona docker?") is None

    def test_first_match_wins(self):
        e1 = ContextMapEntry(patterns=["hola"], task_type="general")
        e2 = ContextMapEntry(patterns=["hola"], task_type="explanation")
        cmap = ContextMap(entries=[e1, e2])
        result = cmap.match("hola")
        assert result.entry.task_type == "general"

    def test_empty_query_returns_none(self):
        entry = ContextMapEntry(patterns=[".*"], task_type="general")
        cmap = ContextMap(entries=[entry])
        assert cmap.match("") is None
        assert cmap.match("   ") is None

    def test_empty_map_returns_none(self):
        cmap = ContextMap()
        assert cmap.match("hola") is None

    def test_explanation_patterns(self):
        entry = ContextMapEntry(
            patterns=["\\bqu[eé] (es|son|significa)\\b"],
            task_type="explanation",
            profile_fragments=["identity", "tech_stack"],
        )
        cmap = ContextMap(entries=[entry])

        assert cmap.match("qué es un decorator?") is not None
        assert cmap.match("que son los microservicios?") is not None
        assert cmap.match("qué significa SOLID?") is not None
        assert cmap.match("como configuro nginx?") is None

    def test_comparison_patterns(self):
        entry = ContextMapEntry(
            patterns=["\\bvs\\.?\\b", "\\bdiferen(cia|ce)\\b.*\\bentre\\b"],
            task_type="explanation",
        )
        cmap = ContextMap(entries=[entry])
        assert cmap.match("python vs javascript") is not None
        assert cmap.match("diferencia entre REST y GraphQL") is not None
        assert cmap.match("diferencia entre arrays y listas") is not None


# ---------------------------------------------------------------------------
# ContextMap.from_file
# ---------------------------------------------------------------------------


class TestContextMapFromFile:
    def test_load_valid_file(self, tmp_path):
        data = {
            "entries": [
                {
                    "patterns": ["^hola"],
                    "task_type": "general",
                    "use_context": False,
                    "profile_fragments": ["identity"],
                },
                {
                    "patterns": ["\\bqu[eé] es\\b"],
                    "task_type": "explanation",
                    "use_context": False,
                    "profile_fragments": ["identity", "tech_stack"],
                },
            ]
        }
        path = tmp_path / "context_map.json"
        path.write_text(json.dumps(data))

        cmap = ContextMap.from_file(path)
        assert len(cmap.entries) == 2
        assert cmap.entries[0].task_type == "general"
        assert cmap.entries[1].profile_fragments == ["identity", "tech_stack"]

    def test_missing_file_returns_empty(self, tmp_path):
        cmap = ContextMap.from_file(tmp_path / "nonexistent.json")
        assert len(cmap.entries) == 0

    def test_invalid_json_returns_empty(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json")
        cmap = ContextMap.from_file(path)
        assert len(cmap.entries) == 0

    def test_entries_without_patterns_skipped(self, tmp_path):
        data = {
            "entries": [
                {"task_type": "general"},  # no patterns
                {"patterns": ["^test"], "task_type": "testing"},
            ]
        }
        path = tmp_path / "map.json"
        path.write_text(json.dumps(data))
        cmap = ContextMap.from_file(path)
        assert len(cmap.entries) == 1

    def test_invalid_fragments_filtered(self, tmp_path):
        data = {
            "entries": [
                {
                    "patterns": ["^test"],
                    "profile_fragments": ["identity", "invalid_fragment", "tech_stack"],
                }
            ]
        }
        path = tmp_path / "map.json"
        path.write_text(json.dumps(data))
        cmap = ContextMap.from_file(path)
        assert cmap.entries[0].profile_fragments == ["identity", "tech_stack"]

    def test_empty_fragments_defaults_to_identity(self, tmp_path):
        data = {"entries": [{"patterns": ["^test"], "profile_fragments": ["bad1", "bad2"]}]}
        path = tmp_path / "map.json"
        path.write_text(json.dumps(data))
        cmap = ContextMap.from_file(path)
        assert cmap.entries[0].profile_fragments == ["identity"]

    def test_loads_real_context_map(self):
        """Verify the actual data/profile/context_map.json is valid."""
        from pathlib import Path

        real_path = Path("data/profile/context_map.json")
        if real_path.exists():
            cmap = ContextMap.from_file(real_path)
            assert len(cmap.entries) > 0
            # Verify all entries have compiled patterns
            for entry in cmap.entries:
                assert len(entry._compiled) > 0


# ---------------------------------------------------------------------------
# VALID_FRAGMENTS
# ---------------------------------------------------------------------------


class TestValidFragments:
    def test_expected_fragments(self):
        assert "identity" in VALID_FRAGMENTS
        assert "tech_stack" in VALID_FRAGMENTS
        assert "patterns" in VALID_FRAGMENTS
        assert "projects" in VALID_FRAGMENTS
        assert "decisions" in VALID_FRAGMENTS
        assert "style" in VALID_FRAGMENTS

    def test_invalid_fragment_not_included(self):
        assert "foo" not in VALID_FRAGMENTS
        assert "metadata" not in VALID_FRAGMENTS


# ---------------------------------------------------------------------------
# Integration: ContextMap with profile fragments
# ---------------------------------------------------------------------------


class TestContextMapIntegration:
    def test_greeting_skips_context(self):
        """Greetings should skip RAG context entirely."""
        entry = ContextMapEntry(
            patterns=["^(hola|hey)\\b"],
            task_type="general",
            use_context=False,
            profile_fragments=["identity"],
        )
        cmap = ContextMap(entries=[entry])

        result = cmap.match("hola, como estas?")
        assert result is not None
        assert result.entry.use_context is False
        assert result.entry.profile_fragments == ["identity"]

    def test_technical_query_no_match(self):
        """Technical queries should fall through to full pipeline."""
        entry = ContextMapEntry(
            patterns=["^(hola|hey)\\b"],
            task_type="general",
        )
        cmap = ContextMap(entries=[entry])
        assert cmap.match("como implemento un decorator en python?") is None

    def test_multiple_entries_coverage(self):
        """Multiple entry types should route correctly."""
        entries = [
            ContextMapEntry(
                patterns=["^(hola|hey)\\b"],
                task_type="general",
                use_context=False,
            ),
            ContextMapEntry(
                patterns=["\\bqu[eé] es\\b"],
                task_type="explanation",
                use_context=False,
                profile_fragments=["identity", "tech_stack"],
            ),
            ContextMapEntry(
                patterns=["\\bc[oó]mo instalo\\b"],
                task_type="devops",
                use_context=False,
                profile_fragments=["identity", "tech_stack"],
            ),
        ]
        cmap = ContextMap(entries=entries)

        r1 = cmap.match("hola!")
        assert r1.entry.task_type == "general"

        r2 = cmap.match("qué es un ORM?")
        assert r2.entry.task_type == "explanation"

        r3 = cmap.match("cómo instalo docker?")
        assert r3.entry.task_type == "devops"

        # Falls through
        assert cmap.match("este test falla con error 500") is None
