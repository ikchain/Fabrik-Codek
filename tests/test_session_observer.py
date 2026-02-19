"""Tests for the session observer."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.flywheel import session_observer


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _make_session_jsonl(pairs: list[tuple[str, str]]) -> str:
    """Helper: create session JSONL from (user_text, assistant_text) pairs."""
    lines = []
    for i, (user_text, assistant_text) in enumerate(pairs):
        user_msg = {
            "type": "user",
            "uuid": f"user-{i}",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": user_text}],
            },
            "timestamp": f"2026-02-16T10:{i:02d}:00+01:00",
        }
        assistant_msg = {
            "type": "assistant",
            "uuid": f"asst-{i}",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
            "timestamp": f"2026-02-16T10:{i:02d}:01+01:00",
        }
        lines.append(json.dumps(user_msg))
        lines.append(json.dumps(assistant_msg))
    return "\n".join(lines) + "\n"


class TestGetProcessedSessions:
    def test_no_marker_file(self, tmp_dir):
        marker = tmp_dir / ".processed"
        with patch.object(session_observer, "PROCESSED_MARKER", marker):
            result = session_observer.get_processed_sessions()
            assert result == set()

    def test_existing_marker_file(self, tmp_dir):
        marker = tmp_dir / ".processed"
        marker.write_text("project1/session1.jsonl\nproject2/session2.jsonl\n")
        with patch.object(session_observer, "PROCESSED_MARKER", marker):
            result = session_observer.get_processed_sessions()
            assert "project1/session1.jsonl" in result
            assert "project2/session2.jsonl" in result
            # strip() removes trailing newline, so split gives exactly 2 entries
            assert len(result) == 2

    def test_empty_marker_file(self, tmp_dir):
        """An empty file returns a set with one empty string due to strip+split."""
        marker = tmp_dir / ".processed"
        marker.write_text("")
        with patch.object(session_observer, "PROCESSED_MARKER", marker):
            result = session_observer.get_processed_sessions()
            # "".strip().split("\n") == [""] -> {""}
            assert result == {""}

    def test_single_entry(self, tmp_dir):
        marker = tmp_dir / ".processed"
        marker.write_text("proj/session.jsonl\n")
        with patch.object(session_observer, "PROCESSED_MARKER", marker):
            result = session_observer.get_processed_sessions()
            assert result == {"proj/session.jsonl"}


class TestMarkSessionProcessed:
    def test_mark_new_session(self, tmp_dir):
        marker = tmp_dir / "metadata" / ".processed"
        with patch.object(session_observer, "PROCESSED_MARKER", marker):
            session_observer.mark_session_processed("proj/session.jsonl")
            assert marker.exists()
            assert "proj/session.jsonl" in marker.read_text()

    def test_mark_multiple_sessions(self, tmp_dir):
        marker = tmp_dir / ".processed"
        with patch.object(session_observer, "PROCESSED_MARKER", marker):
            session_observer.mark_session_processed("a/1.jsonl")
            session_observer.mark_session_processed("b/2.jsonl")
            content = marker.read_text()
            assert "a/1.jsonl" in content
            assert "b/2.jsonl" in content

    def test_creates_parent_dirs(self, tmp_dir):
        marker = tmp_dir / "deep" / "nested" / ".processed"
        with patch.object(session_observer, "PROCESSED_MARKER", marker):
            session_observer.mark_session_processed("x/y.jsonl")
            assert marker.parent.exists()
            assert marker.exists()


class TestExtractConversationPairs:
    def test_basic_pairs(self, tmp_dir):
        session_file = tmp_dir / "session.jsonl"
        session_file.write_text(_make_session_jsonl([
            ("How to use FastAPI?", "FastAPI is a modern web framework..."),
            ("Fix the bug", "I fixed the null pointer issue."),
        ]))
        pairs = list(session_observer.extract_conversation_pairs(session_file))
        assert len(pairs) == 2
        assert pairs[0]["instruction"] == "How to use FastAPI?"
        assert "FastAPI" in pairs[0]["output"]

    def test_content_as_string(self, tmp_dir):
        """Handle content as plain string (not list)."""
        session_file = tmp_dir / "session.jsonl"
        lines = [
            json.dumps({"type": "user", "message": {"role": "user", "content": "Hello"}}),
            json.dumps({"type": "assistant", "message": {"role": "assistant", "content": "Hi there"}}),
        ]
        session_file.write_text("\n".join(lines) + "\n")
        pairs = list(session_observer.extract_conversation_pairs(session_file))
        assert len(pairs) == 1
        assert pairs[0]["instruction"] == "Hello"
        assert pairs[0]["output"] == "Hi there"

    def test_malformed_jsonl_skipped(self, tmp_dir):
        session_file = tmp_dir / "session.jsonl"
        content = _make_session_jsonl([("Valid Q", "Valid A")])
        session_file.write_text("not valid json\n" + content)
        pairs = list(session_observer.extract_conversation_pairs(session_file))
        assert len(pairs) == 1

    def test_empty_session(self, tmp_dir):
        session_file = tmp_dir / "empty.jsonl"
        session_file.write_text("")
        pairs = list(session_observer.extract_conversation_pairs(session_file))
        assert pairs == []

    def test_no_assistant_response(self, tmp_dir):
        """User message without following assistant -> no pair."""
        session_file = tmp_dir / "session.jsonl"
        session_file.write_text(json.dumps({
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": "Hello?"}]},
        }) + "\n")
        pairs = list(session_observer.extract_conversation_pairs(session_file))
        assert pairs == []

    def test_tool_use_messages_skipped(self, tmp_dir):
        """Non user/assistant types are skipped."""
        session_file = tmp_dir / "session.jsonl"
        lines = [
            json.dumps({"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "Q"}]}}),
            json.dumps({"type": "tool_result", "message": {"content": "tool output"}}),
            json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "A"}]}}),
        ]
        session_file.write_text("\n".join(lines) + "\n")
        pairs = list(session_observer.extract_conversation_pairs(session_file))
        assert len(pairs) == 1
        assert pairs[0]["instruction"] == "Q"
        assert pairs[0]["output"] == "A"

    def test_multiple_user_before_assistant(self, tmp_dir):
        """If two user messages come before an assistant, only the last user pairs."""
        session_file = tmp_dir / "session.jsonl"
        lines = [
            json.dumps({"type": "user", "message": {"role": "user", "content": "First question"}}),
            json.dumps({"type": "user", "message": {"role": "user", "content": "Second question"}}),
            json.dumps({"type": "assistant", "message": {"role": "assistant", "content": "Answer to second"}}),
        ]
        session_file.write_text("\n".join(lines) + "\n")
        pairs = list(session_observer.extract_conversation_pairs(session_file))
        assert len(pairs) == 1
        assert pairs[0]["instruction"] == "Second question"

    def test_timestamp_preserved(self, tmp_dir):
        session_file = tmp_dir / "session.jsonl"
        session_file.write_text(_make_session_jsonl([
            ("Question", "Answer"),
        ]))
        pairs = list(session_observer.extract_conversation_pairs(session_file))
        assert len(pairs) == 1
        assert "timestamp" in pairs[0]
        assert pairs[0]["timestamp"] != ""

    def test_empty_content_skipped(self, tmp_dir):
        """Messages with empty content are not added to the message list."""
        session_file = tmp_dir / "session.jsonl"
        lines = [
            json.dumps({"type": "user", "message": {"role": "user", "content": ""}}),
            json.dumps({"type": "assistant", "message": {"role": "assistant", "content": "Orphan answer"}}),
        ]
        session_file.write_text("\n".join(lines) + "\n")
        pairs = list(session_observer.extract_conversation_pairs(session_file))
        # Empty user content is skipped, so assistant has no user to pair with
        assert pairs == []


class TestCategorizeInteraction:
    def test_code_generation_with_code_block(self):
        cat = session_observer.categorize_interaction(
            "Write a function to sort arrays",
            "Here is the code:\n```python\ndef sort_array(arr):\n    return sorted(arr)\n```"
        )
        assert cat == "code_generation"

    def test_code_generation_needs_code_block(self):
        """'Write' keyword without code block in output -> not code_generation."""
        cat = session_observer.categorize_interaction("Write docs", "Here are the docs without code")
        assert cat != "code_generation"

    def test_error_fix(self):
        cat = session_observer.categorize_interaction("Fix the connection error", "Done")
        assert cat == "error_fix"

    def test_error_fix_bug_keyword(self):
        cat = session_observer.categorize_interaction("There is a bug in the login", "Fixed it")
        assert cat == "error_fix"

    def test_refactor(self):
        cat = session_observer.categorize_interaction("Improve the database query", "Optimized")
        assert cat == "refactor"

    def test_refactor_optimize_keyword(self):
        cat = session_observer.categorize_interaction("Optimize the rendering loop", "Done")
        assert cat == "refactor"

    def test_explanation(self):
        cat = session_observer.categorize_interaction("Explain how Docker works", "Docker is...")
        assert cat == "explanation"

    def test_explanation_how_keyword(self):
        cat = session_observer.categorize_interaction("How does JWT authentication work?", "JWT is...")
        assert cat == "explanation"

    def test_code_review(self):
        cat = session_observer.categorize_interaction("Review this code", "Looks good")
        assert cat == "code_review"

    def test_decision(self):
        cat = session_observer.categorize_interaction("Should I use Redis or Memcached?", "Redis is better for...")
        assert cat == "decision"

    def test_general_fallback(self):
        cat = session_observer.categorize_interaction("Hello there", "Hi!")
        assert cat == "general"

    def test_priority_code_gen_over_explanation(self):
        """'Write' with '```' should be code_generation even if 'how' is also present."""
        cat = session_observer.categorize_interaction(
            "Write me a function showing how to sort",
            "Sure:\n```python\nsorted(x)\n```"
        )
        assert cat == "code_generation"

    def test_create_keyword_with_code_block(self):
        cat = session_observer.categorize_interaction(
            "Create a REST endpoint",
            "```python\n@app.get('/api')\ndef endpoint(): pass\n```"
        )
        assert cat == "code_generation"

    def test_spanish_keywords(self):
        assert session_observer.categorize_interaction("Arregla el error", "Listo") == "error_fix"
        assert session_observer.categorize_interaction("Mejora este codigo", "Ok") == "refactor"
        assert session_observer.categorize_interaction("Explica el patron", "Es...") == "explanation"


class TestCalculateQualityScore:
    def test_base_score(self):
        """Base score is 0.5, moderate length output should stay around that."""
        score = session_observer.calculate_quality_score(
            "A question about Python",
            "A reasonable answer of moderate length that is at least a hundred characters long so it avoids penalties for being too short"
        )
        assert 0.0 <= score <= 1.0

    def test_long_output_bonus(self):
        """Output > 500 chars gets +0.1 bonus."""
        short_output = "Word " * 30  # 150 chars, 30 words -> no length penalty, no bonus
        long_output = "Word " * 120  # 600 chars, 120 words -> +0.1 bonus
        short_score = session_observer.calculate_quality_score("Q", short_output)
        long_score = session_observer.calculate_quality_score("Q", long_output)
        assert long_score > short_score

    def test_very_long_output_extra_bonus(self):
        """Output > 1000 chars gets +0.1 + 0.1 = +0.2 bonus."""
        # Need enough words (>= 10) to avoid the -0.3 penalty
        output = "Word " * 250  # 1250 chars, 250 words
        score = session_observer.calculate_quality_score("Q", output)
        # 0.5 base + 0.1 (>500) + 0.1 (>1000) = 0.7
        assert score >= 0.7

    def test_code_presence_bonus(self):
        no_code = session_observer.calculate_quality_score(
            "Q", "Just text without any code blocks " * 5
        )
        with_code = session_observer.calculate_quality_score(
            "Q", "Here:\n```python\ncode\n```\n" + "explanation " * 10
        )
        assert with_code > no_code

    def test_structure_bonus(self):
        flat = session_observer.calculate_quality_score(
            "Q", "Just a paragraph of text without any structure markers at all " * 3
        )
        structured = session_observer.calculate_quality_score(
            "Q", "- item1\n- item2\n**bold**\n" + "explanation " * 20
        )
        assert structured > flat

    def test_short_output_penalty(self):
        """Output < 100 chars and < 10 words gets heavy penalties."""
        score = session_observer.calculate_quality_score("Q", "Very short answer")
        # 0.5 - 0.2 (< 100 chars) - 0.3 (< 10 words) = 0.0
        assert score < 0.5

    def test_single_word_penalty(self):
        """Single word: base 0.5 - 0.2 (short) - 0.3 (few words) = 0.0."""
        score = session_observer.calculate_quality_score("Q", "Yes")
        assert score <= 0.1

    def test_score_lower_bound(self):
        """Score should never go below 0."""
        score = session_observer.calculate_quality_score("Q", "No")
        assert score >= 0.0

    def test_score_upper_bound(self):
        """Score should never exceed 1.0."""
        score = session_observer.calculate_quality_score(
            "Q", "```code```\n" + "A " * 600 + "\n- list\n**bold**"
        )
        assert score <= 1.0

    def test_all_bonuses_combined(self):
        """Long output with code and structure gets maximum bonuses."""
        output = "```python\ncode\n```\n- item\n**bold**\n" + "Word " * 250
        score = session_observer.calculate_quality_score("Q", output)
        # 0.5 + 0.1 + 0.1 + 0.15 + 0.05 = 0.9
        assert score == pytest.approx(0.9, abs=0.01)


class TestProcessSession:
    def test_process_extracts_pairs(self, tmp_dir):
        session_file = tmp_dir / "session.jsonl"
        long_output = "This is a detailed explanation of how Docker works with containers. " * 10
        session_file.write_text(_make_session_jsonl([
            ("How does Docker work?", long_output),
        ]))
        pairs = session_observer.process_session(session_file, min_quality=0.3)
        assert len(pairs) >= 1
        assert pairs[0]["category"] == "explanation"
        assert "id" in pairs[0]
        assert pairs[0]["source"] == "claude-code-session"

    def test_skips_short_instructions(self, tmp_dir):
        """Instructions < 10 chars are skipped."""
        session_file = tmp_dir / "session.jsonl"
        session_file.write_text(_make_session_jsonl([
            ("Hi", "Hello there, this is a moderately long response that should pass quality"),
        ]))
        pairs = session_observer.process_session(session_file, min_quality=0.0)
        assert pairs == []

    def test_skips_short_outputs(self, tmp_dir):
        """Outputs < 20 chars are skipped."""
        session_file = tmp_dir / "session.jsonl"
        session_file.write_text(_make_session_jsonl([
            ("Tell me about Python programming", "Short"),
        ]))
        pairs = session_observer.process_session(session_file, min_quality=0.0)
        assert pairs == []

    def test_skips_short_interactions(self, tmp_dir):
        session_file = tmp_dir / "session.jsonl"
        session_file.write_text(_make_session_jsonl([
            ("Hi", "Hello"),  # Both too short
        ]))
        pairs = session_observer.process_session(session_file)
        assert pairs == []

    def test_quality_threshold(self, tmp_dir):
        session_file = tmp_dir / "session.jsonl"
        # Output > 20 chars, instruction > 10 chars, but low quality
        session_file.write_text(_make_session_jsonl([
            ("Fix the error in the code please", "ok done, it is fixed now"),
        ]))
        pairs = session_observer.process_session(session_file, min_quality=0.5)
        assert pairs == []

    def test_training_pair_format(self, tmp_dir):
        session_file = tmp_dir / "session.jsonl"
        long_output = (
            "Here is a comprehensive explanation:\n"
            "```python\ndef example():\n    pass\n```\n"
            + "Details about this implementation. " * 30
        )
        session_file.write_text(_make_session_jsonl([
            ("Write a Python function for sorting", long_output),
        ]))
        pairs = session_observer.process_session(session_file, min_quality=0.3)
        assert len(pairs) >= 1
        pair = pairs[0]
        assert "id" in pair
        assert "instruction" in pair
        assert "output" in pair
        assert "category" in pair
        assert "quality_score" in pair
        assert "source_file" in pair
        assert "extracted_at" in pair
        assert pair["source"] == "claude-code-session"
        assert pair["input"] == ""

    def test_id_is_deterministic(self, tmp_dir):
        """Same content should produce same ID."""
        session_file = tmp_dir / "session.jsonl"
        output_text = "Detailed answer about Python programming concepts and patterns " * 5
        session_file.write_text(_make_session_jsonl([
            ("Tell me about Python patterns", output_text),
        ]))
        pairs1 = session_observer.process_session(session_file, min_quality=0.3)
        pairs2 = session_observer.process_session(session_file, min_quality=0.3)
        assert pairs1[0]["id"] == pairs2[0]["id"]

    def test_multiple_pairs_extracted(self, tmp_dir):
        session_file = tmp_dir / "session.jsonl"
        long_a1 = "Docker uses containerization to isolate applications. " * 10
        long_a2 = "Kubernetes orchestrates Docker containers at scale. " * 10
        session_file.write_text(_make_session_jsonl([
            ("Explain how Docker works in detail", long_a1),
            ("Explain how Kubernetes works in detail", long_a2),
        ]))
        pairs = session_observer.process_session(session_file, min_quality=0.3)
        assert len(pairs) == 2


class TestProcessAllSessions:
    def test_process_all(self, tmp_dir):
        projects_dir = tmp_dir / "projects"
        project1 = projects_dir / "project1"
        project1.mkdir(parents=True)

        long_output = "Detailed explanation of architecture patterns with examples. " * 10
        session_file = project1 / "session1.jsonl"
        session_file.write_text(_make_session_jsonl([
            ("Explain hexagonal architecture", long_output),
        ]))

        marker = tmp_dir / ".processed"
        sessions_out = tmp_dir / "sessions"
        training_out = tmp_dir / "training"

        with patch.object(session_observer, "CLAUDE_PROJECTS_DIR", projects_dir), \
             patch.object(session_observer, "PROCESSED_MARKER", marker), \
             patch.object(session_observer, "SESSIONS_OUTPUT", sessions_out), \
             patch.object(session_observer, "TRAINING_OUTPUT", training_out):

            stats = session_observer.process_all_sessions(min_quality=0.3)
            assert stats["sessions_processed"] >= 1
            assert stats["pairs_extracted"] >= 1
            assert "output_file" in stats

    def test_skips_already_processed(self, tmp_dir):
        projects_dir = tmp_dir / "projects"
        project1 = projects_dir / "project1"
        project1.mkdir(parents=True)

        long_output = "Detailed explanation. " * 20
        session_file = project1 / "session1.jsonl"
        session_file.write_text(_make_session_jsonl([
            ("Explain something useful and interesting", long_output),
        ]))

        marker = tmp_dir / ".processed"
        marker.write_text("project1/session1.jsonl\n")
        sessions_out = tmp_dir / "sessions"
        training_out = tmp_dir / "training"

        with patch.object(session_observer, "CLAUDE_PROJECTS_DIR", projects_dir), \
             patch.object(session_observer, "PROCESSED_MARKER", marker), \
             patch.object(session_observer, "SESSIONS_OUTPUT", sessions_out), \
             patch.object(session_observer, "TRAINING_OUTPUT", training_out):

            stats = session_observer.process_all_sessions()
            assert stats["sessions_processed"] == 0
            assert stats["pairs_extracted"] == 0

    def test_creates_output_directories(self, tmp_dir):
        projects_dir = tmp_dir / "projects"
        projects_dir.mkdir()

        marker = tmp_dir / ".processed"
        sessions_out = tmp_dir / "new_sessions"
        training_out = tmp_dir / "new_training"

        with patch.object(session_observer, "CLAUDE_PROJECTS_DIR", projects_dir), \
             patch.object(session_observer, "PROCESSED_MARKER", marker), \
             patch.object(session_observer, "SESSIONS_OUTPUT", sessions_out), \
             patch.object(session_observer, "TRAINING_OUTPUT", training_out):

            session_observer.process_all_sessions()
            assert sessions_out.exists()
            assert training_out.exists()

    def test_marks_session_even_without_pairs(self, tmp_dir):
        """Sessions with no quality pairs are still marked as processed."""
        projects_dir = tmp_dir / "projects"
        project1 = projects_dir / "project1"
        project1.mkdir(parents=True)

        session_file = project1 / "session1.jsonl"
        session_file.write_text(_make_session_jsonl([
            ("Hi", "Hello"),  # Too short -> no pairs
        ]))

        marker = tmp_dir / ".processed"
        sessions_out = tmp_dir / "sessions"
        training_out = tmp_dir / "training"

        with patch.object(session_observer, "CLAUDE_PROJECTS_DIR", projects_dir), \
             patch.object(session_observer, "PROCESSED_MARKER", marker), \
             patch.object(session_observer, "SESSIONS_OUTPUT", sessions_out), \
             patch.object(session_observer, "TRAINING_OUTPUT", training_out):

            stats = session_observer.process_all_sessions()
            assert stats["sessions_processed"] == 0  # No pairs extracted
            assert stats["pairs_extracted"] == 0
            # But session was still marked as processed
            assert marker.exists()
            assert "project1/session1.jsonl" in marker.read_text()

    def test_writes_training_jsonl(self, tmp_dir):
        """Verify the output file contains valid JSONL training pairs."""
        projects_dir = tmp_dir / "projects"
        project1 = projects_dir / "project1"
        project1.mkdir(parents=True)

        long_output = "Detailed explanation with many words for quality. " * 15
        session_file = project1 / "session1.jsonl"
        session_file.write_text(_make_session_jsonl([
            ("Explain hexagonal architecture patterns", long_output),
        ]))

        marker = tmp_dir / ".processed"
        sessions_out = tmp_dir / "sessions"
        training_out = tmp_dir / "training"

        with patch.object(session_observer, "CLAUDE_PROJECTS_DIR", projects_dir), \
             patch.object(session_observer, "PROCESSED_MARKER", marker), \
             patch.object(session_observer, "SESSIONS_OUTPUT", sessions_out), \
             patch.object(session_observer, "TRAINING_OUTPUT", training_out):

            stats = session_observer.process_all_sessions(min_quality=0.3)
            output_path = Path(stats["output_file"])
            assert output_path.exists()
            with open(output_path) as f:
                lines = f.readlines()
            assert len(lines) >= 1
            pair = json.loads(lines[0])
            assert "instruction" in pair
            assert "output" in pair

    def test_skips_non_directory_children(self, tmp_dir):
        """Files directly under CLAUDE_PROJECTS_DIR are skipped."""
        projects_dir = tmp_dir / "projects"
        projects_dir.mkdir(parents=True)
        # Create a file, not a directory
        (projects_dir / "stray_file.txt").write_text("not a project dir")

        marker = tmp_dir / ".processed"
        sessions_out = tmp_dir / "sessions"
        training_out = tmp_dir / "training"

        with patch.object(session_observer, "CLAUDE_PROJECTS_DIR", projects_dir), \
             patch.object(session_observer, "PROCESSED_MARKER", marker), \
             patch.object(session_observer, "SESSIONS_OUTPUT", sessions_out), \
             patch.object(session_observer, "TRAINING_OUTPUT", training_out):

            stats = session_observer.process_all_sessions()
            assert stats["sessions_processed"] == 0

    def test_by_category_stats(self, tmp_dir):
        projects_dir = tmp_dir / "projects"
        project1 = projects_dir / "project1"
        project1.mkdir(parents=True)

        explanation_output = "Docker uses containerization to package applications. " * 15
        fix_output = "I fixed the null pointer by adding a guard clause. " * 15
        session_file = project1 / "session1.jsonl"
        session_file.write_text(_make_session_jsonl([
            ("Explain how Docker containers work in detail", explanation_output),
            ("Fix the null pointer error in the login module", fix_output),
        ]))

        marker = tmp_dir / ".processed"
        sessions_out = tmp_dir / "sessions"
        training_out = tmp_dir / "training"

        with patch.object(session_observer, "CLAUDE_PROJECTS_DIR", projects_dir), \
             patch.object(session_observer, "PROCESSED_MARKER", marker), \
             patch.object(session_observer, "SESSIONS_OUTPUT", sessions_out), \
             patch.object(session_observer, "TRAINING_OUTPUT", training_out):

            stats = session_observer.process_all_sessions(min_quality=0.3)
            assert stats["pairs_extracted"] >= 2
            assert "by_category" in stats
            assert len(stats["by_category"]) >= 1


class TestGetStats:
    def test_stats_with_training_files(self, tmp_dir):
        training_out = tmp_dir / "training"
        training_out.mkdir(parents=True)
        tf = training_out / "claude-sessions_20260216.jsonl"
        with open(tf, "w") as f:
            f.write(json.dumps({"instruction": "Q1", "output": "A1"}) + "\n")
            f.write(json.dumps({"instruction": "Q2", "output": "A2"}) + "\n")

        marker = tmp_dir / ".processed"
        marker.write_text("proj/s1.jsonl\nproj/s2.jsonl\n")

        with patch.object(session_observer, "TRAINING_OUTPUT", training_out), \
             patch.object(session_observer, "PROCESSED_MARKER", marker):
            stats = session_observer.get_stats()
            assert stats["sessions_processed"] == 2
            assert stats["total_training_pairs"] == 2

    def test_stats_no_files(self, tmp_dir):
        training_out = tmp_dir / "training"
        training_out.mkdir(parents=True)
        marker = tmp_dir / ".processed"

        with patch.object(session_observer, "TRAINING_OUTPUT", training_out), \
             patch.object(session_observer, "PROCESSED_MARKER", marker):
            stats = session_observer.get_stats()
            assert stats["sessions_processed"] == 0
            assert stats["total_training_pairs"] == 0

    def test_stats_multiple_training_files(self, tmp_dir):
        training_out = tmp_dir / "training"
        training_out.mkdir(parents=True)

        tf1 = training_out / "claude-sessions_20260216_100000.jsonl"
        with open(tf1, "w") as f:
            f.write(json.dumps({"instruction": "Q1"}) + "\n")
            f.write(json.dumps({"instruction": "Q2"}) + "\n")

        tf2 = training_out / "claude-sessions_20260217_100000.jsonl"
        with open(tf2, "w") as f:
            f.write(json.dumps({"instruction": "Q3"}) + "\n")

        marker = tmp_dir / ".processed"
        marker.write_text("a/1.jsonl\n")

        with patch.object(session_observer, "TRAINING_OUTPUT", training_out), \
             patch.object(session_observer, "PROCESSED_MARKER", marker):
            stats = session_observer.get_stats()
            assert stats["sessions_processed"] == 1
            assert stats["total_training_pairs"] == 3

    def test_stats_includes_training_dir(self, tmp_dir):
        training_out = tmp_dir / "training"
        training_out.mkdir(parents=True)
        marker = tmp_dir / ".processed"

        with patch.object(session_observer, "TRAINING_OUTPUT", training_out), \
             patch.object(session_observer, "PROCESSED_MARKER", marker):
            stats = session_observer.get_stats()
            assert "training_dir" in stats
            assert stats["training_dir"] == str(training_out)

    def test_stats_ignores_non_matching_files(self, tmp_dir):
        """Only files matching claude-sessions_*.jsonl are counted."""
        training_out = tmp_dir / "training"
        training_out.mkdir(parents=True)

        # This file does NOT match the glob pattern
        other = training_out / "other_data.jsonl"
        with open(other, "w") as f:
            f.write(json.dumps({"instruction": "Q1"}) + "\n")

        marker = tmp_dir / ".processed"

        with patch.object(session_observer, "TRAINING_OUTPUT", training_out), \
             patch.object(session_observer, "PROCESSED_MARKER", marker):
            stats = session_observer.get_stats()
            assert stats["total_training_pairs"] == 0
