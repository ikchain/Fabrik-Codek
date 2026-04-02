"""Tests for context compaction."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.compaction import (
    COMPACTED_PREFIX,
    TASK_COMPACTION_THRESHOLDS,
    CompactionThresholds,
    compact_if_needed,
    estimate_tokens,
    get_thresholds,
    snip_compact,
    summary_compact,
)

# ---------------------------------------------------------------------------
# TestCompactionThresholds
# ---------------------------------------------------------------------------


class TestCompactionThresholds:
    def test_defaults(self):
        t = CompactionThresholds()
        assert t.soft_limit == 3500
        assert t.hard_limit == 5000
        assert t.emergency_limit == 6000
        assert t.keep_turns == 6
        assert t.max_bullets == 8

    def test_frozen(self):
        t = CompactionThresholds()
        with pytest.raises(AttributeError):
            t.soft_limit = 9999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestGetThresholds
# ---------------------------------------------------------------------------


class TestGetThresholds:
    def test_known_task_type(self):
        t = get_thresholds("debugging")
        assert t.soft_limit == 4500
        assert t.keep_turns == 8
        assert t.max_bullets == 10

    def test_unknown_task_type_returns_defaults(self):
        t = get_thresholds("unknown_type")
        assert t.soft_limit == 3500

    def test_none_returns_defaults(self):
        t = get_thresholds(None)
        assert t.soft_limit == 3500

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("FABRIK_COMPACT_SOFT_LIMIT", "2000")
        monkeypatch.setenv("FABRIK_COMPACT_HARD_LIMIT", "3000")
        t = get_thresholds(None)
        assert t.soft_limit == 2000
        assert t.hard_limit == 3000
        assert t.emergency_limit == 4000  # hard + 1000

    def test_all_task_types_present(self):
        expected = {"debugging", "code_review", "architecture", "general", "explanation"}
        assert set(TASK_COMPACTION_THRESHOLDS.keys()) == expected


# ---------------------------------------------------------------------------
# TestEstimateTokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens([]) == 0

    def test_known_text(self):
        msg = [{"role": "user", "content": "a" * 400}]
        assert estimate_tokens(msg) == 100  # 400 // 4

    def test_multiple_messages(self):
        msgs = [
            {"role": "user", "content": "a" * 200},
            {"role": "assistant", "content": "b" * 400},
        ]
        assert estimate_tokens(msgs) == 150  # 50 + 100

    def test_system_excluded(self):
        msgs = [
            {"role": "system", "content": "x" * 1000},
            {"role": "user", "content": "a" * 200},
        ]
        assert estimate_tokens(msgs) == 50  # only user counted


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_turn(i: int) -> list[dict]:
    """Create a user+assistant pair for turn i."""
    return [
        {"role": "user", "content": f"question {i}"},
        {"role": "assistant", "content": f"answer {i}"},
    ]


def _make_chat(system: str, num_turns: int) -> list[dict]:
    """Build a chat with system + N turn pairs."""
    msgs = [{"role": "system", "content": system}]
    for i in range(num_turns):
        msgs.extend(_make_turn(i))
    return msgs


# ---------------------------------------------------------------------------
# TestSnipCompact
# ---------------------------------------------------------------------------


class TestSnipCompact:
    def test_fewer_turns_than_keep_unchanged(self):
        msgs = _make_chat("sys", 3)  # 3 turns < keep_turns=6
        result = snip_compact(msgs, keep_turns=6)
        assert result == msgs

    def test_exactly_at_keep_turns_unchanged(self):
        msgs = _make_chat("sys", 6)  # exactly 6 turns
        result = snip_compact(msgs, keep_turns=6)
        assert result == msgs

    def test_drops_middle_keeps_system_and_tail(self):
        msgs = _make_chat("sys", 10)  # 10 turns, keep last 4
        result = snip_compact(msgs, keep_turns=4)
        assert result[0] == {"role": "system", "content": "sys"}
        # Last 4 turns = 8 messages, starting from turn 6
        assert len(result) == 1 + 8  # system + 4 pairs
        assert result[1]["content"] == "question 6"
        assert result[-1]["content"] == "answer 9"

    def test_no_system_message(self):
        msgs = []
        for i in range(5):
            msgs.extend(_make_turn(i))
        result = snip_compact(msgs, keep_turns=2)
        assert len(result) == 4  # 2 pairs
        assert result[0]["content"] == "question 3"

    def test_preserves_message_order(self):
        msgs = _make_chat("sys", 8)
        result = snip_compact(msgs, keep_turns=3)
        roles = [m["role"] for m in result]
        assert roles == ["system", "user", "assistant", "user", "assistant", "user", "assistant"]

    def test_existing_compacted_message_dropped(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "system", "content": f"{COMPACTED_PREFIX} — prior 4 turns]\n- old stuff"},
            *_make_turn(0),
            *_make_turn(1),
            *_make_turn(2),
            *_make_turn(3),
            *_make_turn(4),
        ]
        result = snip_compact(msgs, keep_turns=2)
        # System kept, [COMPACTED] dropped (it's in history), last 2 turns kept
        assert result[0] == {"role": "system", "content": "sys"}
        assert not any(COMPACTED_PREFIX in m.get("content", "") for m in result)
        assert len(result) == 1 + 4  # system + 2 pairs


# ---------------------------------------------------------------------------
# Mock LLM client
# ---------------------------------------------------------------------------


def _mock_client(summary_text: str = "- GOAL: testing\n- STATUS: in progress"):
    """Create a mock LLMClient that returns a fixed summary."""
    client = AsyncMock()
    response = MagicMock()
    response.content = summary_text
    client.chat = AsyncMock(return_value=response)
    return client


def _failing_client():
    """Create a mock LLMClient that raises on chat."""
    client = AsyncMock()
    client.chat = AsyncMock(side_effect=RuntimeError("Ollama down"))
    return client


# ---------------------------------------------------------------------------
# TestSummaryCompact
# ---------------------------------------------------------------------------


class TestSummaryCompact:
    @pytest.mark.asyncio
    async def test_generates_compacted_message(self):
        msgs = _make_chat("sys", 10)
        client = _mock_client("- GOAL: fix auth\n- STATUS: pending")
        result = await summary_compact(msgs, client, keep_turns=3)
        # Structure: system + [COMPACTED] + 3 pairs (6 msgs)
        assert result[0] == {"role": "system", "content": "sys"}
        assert result[1]["role"] == "system"
        assert COMPACTED_PREFIX in result[1]["content"]
        assert "- GOAL: fix auth" in result[1]["content"]
        assert len(result) == 1 + 1 + 6  # sys + compacted + 3 pairs

    @pytest.mark.asyncio
    async def test_code_blocks_stripped_from_input(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "check this:\n```python\nprint('hi')\n```\nplease"},
            {"role": "assistant", "content": "```\nresult\n```\ndone"},
            *_make_turn(1),
            *_make_turn(2),
        ]
        client = _mock_client()
        await summary_compact(msgs, client, keep_turns=1)
        # Verify the input to the LLM call doesn't contain code blocks
        call_args = client.chat.call_args[0][0]  # first positional arg (messages)
        user_content = call_args[1]["content"]
        assert "```" not in user_content
        assert "please" in user_content  # prose preserved

    @pytest.mark.asyncio
    async def test_old_compacted_absorbed_in_summary(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "system", "content": f"{COMPACTED_PREFIX} — prior 4 turns]\n- old context"},
            *_make_turn(0),
            *_make_turn(1),
            *_make_turn(2),
            *_make_turn(3),
        ]
        client = _mock_client("- GOAL: merged\n- STATUS: done")
        result = await summary_compact(msgs, client, keep_turns=2)
        # Old [COMPACTED] should be in the summary input
        call_args = client.chat.call_args[0][0]
        user_content = call_args[1]["content"]
        assert "old context" in user_content
        # Result has new [COMPACTED], not the old one
        compacted_msgs = [m for m in result if COMPACTED_PREFIX in m.get("content", "")]
        assert len(compacted_msgs) == 1
        assert "merged" in compacted_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_llm_call_uses_temperature_zero(self):
        msgs = _make_chat("sys", 10)
        client = _mock_client()
        await summary_compact(msgs, client, keep_turns=3)
        _, kwargs = client.chat.call_args
        assert kwargs.get("temperature") == 0

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_snip(self):
        msgs = _make_chat("sys", 10)
        client = _failing_client()
        result = await summary_compact(msgs, client, keep_turns=3)
        # Should fallback to snip: system + last 3 pairs (6 msgs), no [COMPACTED]
        assert result[0] == {"role": "system", "content": "sys"}
        assert not any(COMPACTED_PREFIX in m.get("content", "") for m in result)
        assert len(result) == 1 + 6

    @pytest.mark.asyncio
    async def test_fewer_turns_than_keep_unchanged(self):
        msgs = _make_chat("sys", 3)
        client = _mock_client()
        result = await summary_compact(msgs, client, keep_turns=6)
        assert result == msgs
        client.chat.assert_not_called()


# ---------------------------------------------------------------------------
# TestCompactIfNeeded
# ---------------------------------------------------------------------------


def _make_large_chat(system: str, num_turns: int, content_size: int = 200) -> list[dict]:
    """Build a chat with large messages to control token estimation."""
    msgs = [{"role": "system", "content": system}]
    for i in range(num_turns):
        msgs.append({"role": "user", "content": f"q{i} " + "x" * content_size})
        msgs.append({"role": "assistant", "content": f"a{i} " + "y" * content_size})
    return msgs


class TestCompactIfNeeded:
    @pytest.mark.asyncio
    async def test_below_soft_noop(self):
        # Small chat, well under any threshold
        msgs = _make_chat("sys", 3)
        client = _mock_client()
        t = CompactionThresholds(soft_limit=50000)
        result, action = await compact_if_needed(msgs, client, t)
        assert action is None
        assert result is msgs  # same reference, untouched

    @pytest.mark.asyncio
    async def test_between_soft_and_hard_snips(self):
        # Enough tokens to exceed soft but not hard
        msgs = _make_large_chat("sys", 20, content_size=300)
        tokens = estimate_tokens(msgs)
        t = CompactionThresholds(
            soft_limit=tokens - 100,
            hard_limit=tokens + 5000,
            emergency_limit=tokens + 6000,
            keep_turns=3,
        )
        client = _mock_client()
        result, action = await compact_if_needed(msgs, client, t)
        assert action == "snip"
        # Should have system + last 3 pairs
        assert result[0]["role"] == "system"
        assert len(result) == 1 + 6
        client.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_above_hard_summarizes(self):
        msgs = _make_large_chat("sys", 20, content_size=300)
        tokens = estimate_tokens(msgs)
        t = CompactionThresholds(
            soft_limit=tokens - 200,
            hard_limit=tokens - 100,
            emergency_limit=tokens + 5000,
            keep_turns=3,
        )
        client = _mock_client("- GOAL: testing")
        result, action = await compact_if_needed(msgs, client, t)
        assert action == "summary"
        assert any(COMPACTED_PREFIX in m.get("content", "") for m in result)
        client.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_emergency_drops_oldest(self):
        # Make summary return something huge so emergency triggers
        huge_summary = "- GOAL: " + "z" * 30000  # ~7500 tokens
        msgs = _make_large_chat("sys", 20, content_size=300)
        tokens = estimate_tokens(msgs)
        t = CompactionThresholds(
            soft_limit=tokens - 200,
            hard_limit=tokens - 100,
            emergency_limit=100,  # very low to force emergency
            keep_turns=3,
        )
        client = _mock_client(huge_summary)
        result, action = await compact_if_needed(msgs, client, t)
        assert action == "emergency"
        # Should have dropped messages to fit
        assert estimate_tokens(result) <= t.emergency_limit or len(result) <= 2

    @pytest.mark.asyncio
    async def test_custom_thresholds_respected(self):
        msgs = _make_large_chat("sys", 5, content_size=100)
        tokens = estimate_tokens(msgs)
        # Set soft below current tokens
        t = CompactionThresholds(
            soft_limit=tokens - 50,
            hard_limit=tokens + 5000,
            emergency_limit=tokens + 6000,
            keep_turns=2,
        )
        client = _mock_client()
        result, action = await compact_if_needed(msgs, client, t)
        assert action == "snip"
        assert len(result) == 1 + 4  # sys + 2 pairs

    @pytest.mark.asyncio
    async def test_returns_new_list(self):
        msgs = _make_large_chat("sys", 20, content_size=300)
        tokens = estimate_tokens(msgs)
        t = CompactionThresholds(
            soft_limit=tokens - 100,
            hard_limit=tokens + 5000,
            emergency_limit=tokens + 6000,
            keep_turns=3,
        )
        client = _mock_client()
        result, _ = await compact_if_needed(msgs, client, t)
        assert result is not msgs

    @pytest.mark.asyncio
    async def test_default_thresholds_when_none(self):
        msgs = _make_chat("sys", 2)  # small, under any threshold
        client = _mock_client()
        result, action = await compact_if_needed(msgs, client, None)
        assert action is None
