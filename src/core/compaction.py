"""Context compaction for chat sessions.

3-layer compaction adapted from Claude Code's 6-layer system for 7B models:
  1. snip_compact — drop middle messages, keep system + last N turn pairs (zero cost)
  2. summary_compact — LLM summarizes discarded portion (1 Ollama call, temp=0)
  3. emergency — drop oldest messages until under hard cap
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

COMPACTED_PREFIX = "[COMPACTED"


@dataclass(frozen=True)
class CompactionThresholds:
    """Token thresholds for compaction triggers.

    All limits refer to user+assistant history tokens only (system excluded).
    """

    soft_limit: int = 3500
    hard_limit: int = 5000
    emergency_limit: int = 6000
    keep_turns: int = 6  # pairs (1 turn = 1 user + 1 assistant message)
    max_bullets: int = 8


# -- Per-task thresholds (aligned with Context Gate always_inject_tasks) ------

TASK_COMPACTION_THRESHOLDS: dict[str, CompactionThresholds] = {
    "debugging": CompactionThresholds(4500, 6000, 7000, 8, 10),
    "code_review": CompactionThresholds(3500, 5000, 6000, 6, 8),
    "architecture": CompactionThresholds(3500, 5000, 6000, 6, 8),
    "general": CompactionThresholds(2500, 4000, 5000, 4, 6),
    "explanation": CompactionThresholds(2500, 4000, 5000, 4, 6),
}


def _default_thresholds() -> CompactionThresholds:
    """Build default thresholds, respecting env var overrides."""
    soft = int(os.environ.get("FABRIK_COMPACT_SOFT_LIMIT", 3500))
    hard = int(os.environ.get("FABRIK_COMPACT_HARD_LIMIT", 5000))
    return CompactionThresholds(soft_limit=soft, hard_limit=hard, emergency_limit=hard + 1000)


def get_thresholds(task_type: str | None = None) -> CompactionThresholds:
    """Return CompactionThresholds for a given task_type."""
    if task_type and task_type in TASK_COMPACTION_THRESHOLDS:
        return TASK_COMPACTION_THRESHOLDS[task_type]
    return _default_thresholds()


# -- Token estimation ---------------------------------------------------------


def estimate_tokens(messages: list[dict]) -> int:
    """Estimate token count for user+assistant messages only.

    System messages are excluded — they have their own budget.
    Uses len(content) // 4 (~3.5 chars/token for mixed EN/ES + code).
    """
    total = 0
    for msg in messages:
        if msg.get("role") in ("user", "assistant"):
            content = msg.get("content", "")
            total += len(content) // 4
    return total


# -- Helpers ------------------------------------------------------------------


def _is_system(msg: dict) -> bool:
    return msg.get("role") == "system"


def _is_compacted(msg: dict) -> bool:
    return _is_system(msg) and msg.get("content", "").startswith(COMPACTED_PREFIX)


def _split_messages(messages: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Split messages into (system_msgs, middle, tail_turn_pairs).

    Note: this does NOT split by keep_turns — caller decides how many to keep.
    Returns (system_messages, history_messages) where system includes all
    system-role messages that are NOT [COMPACTED].
    """
    system_msgs = []
    history = []
    for msg in messages:
        if _is_system(msg) and not _is_compacted(msg):
            system_msgs.append(msg)
        else:
            history.append(msg)
    return system_msgs, history


def _extract_tail_turns(history: list[dict], keep_turns: int) -> tuple[list[dict], list[dict]]:
    """Split history into (discardable, preserved_tail).

    A turn = 1 user + 1 assistant message pair. We count from the end.
    Partial turns (trailing user without assistant) are preserved.
    """
    if keep_turns <= 0:
        return history, []

    # Count pairs from the end
    msgs_to_keep = keep_turns * 2

    # Check for trailing partial turn (user without assistant response)
    if history and history[-1].get("role") == "user":
        msgs_to_keep += 1

    if msgs_to_keep >= len(history):
        return [], history

    split_point = len(history) - msgs_to_keep
    return history[:split_point], history[split_point:]


_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)


def _strip_code_blocks(text: str) -> str:
    """Remove fenced code blocks from text."""
    return _CODE_BLOCK_RE.sub("", text).strip()


# -- Layer 1: snip_compact ---------------------------------------------------


def snip_compact(messages: list[dict], keep_turns: int = 6) -> list[dict]:
    """Drop middle messages, keep system + last N turn pairs.

    Existing [COMPACTED] messages in the middle are dropped (they'll be
    re-summarized by summary_compact if called later).
    Returns a new list — never mutates the original.
    """
    system_msgs, history = _split_messages(messages)
    discardable, tail = _extract_tail_turns(history, keep_turns)

    if not discardable:
        return list(messages)  # nothing to snip

    dropped = len(discardable)
    logger.info("snip_compact: dropped=%d kept_turns=%d", dropped, keep_turns)

    return system_msgs + tail


# -- Layer 2: summary_compact ------------------------------------------------

_SUMMARY_PROMPT = """\
Summarize the following conversation excerpt into key facts.
Output ONLY a bullet list with these categories:
- GOAL: Main task or question the user is working on
- FILES: File paths or modules mentioned
- ERRORS: Errors or issues identified
- ATTEMPTS: Solutions tried (one line each)
- STATUS: Where the conversation left off

Rules:
- Maximum {max_bullets} bullets total
- Skip categories with no information
- Do NOT include code blocks
- Do NOT invent information not present in the conversation
- Be specific: use exact names, paths, error messages
- Preserve the language of the original conversation
"""


def _build_summary_input(discardable: list[dict], max_input_tokens: int = 3000) -> str:
    """Build text input for the summarizer from discardable messages.

    Strips code blocks. Caps at max_input_tokens (estimated).
    """
    parts = []
    for msg in discardable:
        role = msg.get("role", "unknown")
        content = _strip_code_blocks(msg.get("content", ""))
        if content:
            parts.append(f"{role}: {content}")

    text = "\n\n".join(parts)

    # Cap at estimated token limit
    max_chars = max_input_tokens * 4
    if len(text) > max_chars:
        text = text[:max_chars]

    return text


async def summary_compact(
    messages: list[dict],
    client,
    keep_turns: int = 6,
    max_summary_tokens: int = 400,
    max_bullets: int = 8,
) -> list[dict]:
    """Summarize discarded portion with structured prompt (temp=0).

    Old [COMPACTED] messages are included in summary input (absorbed).
    Falls back to snip_compact on LLM failure.

    Parameters
    ----------
    client:
        LLMClient instance (async context manager already entered).
    """
    system_msgs, history = _split_messages(messages)
    discardable, tail = _extract_tail_turns(history, keep_turns)

    if not discardable:
        return list(messages)

    # Count turns being compacted
    turn_count = len(discardable) // 2

    # Build summary input from discardable messages
    summary_input = _build_summary_input(discardable)

    if not summary_input.strip():
        # Nothing meaningful to summarize — just snip
        return system_msgs + tail

    prompt = _SUMMARY_PROMPT.format(max_bullets=max_bullets)
    summary_messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": summary_input},
    ]

    try:
        response = await client.chat(summary_messages, temperature=0)
        summary_text = response.content.strip()
    except Exception:
        logger.warning("summary_compact_llm_failed, falling back to snip")
        return snip_compact(messages, keep_turns)

    if not summary_text:
        return snip_compact(messages, keep_turns)

    compacted_msg = {
        "role": "system",
        "content": f"{COMPACTED_PREFIX} — prior {turn_count} turns]\n{summary_text}",
    }

    logger.info(
        "summary_compact: turns=%d summary_len=%d kept=%d",
        turn_count,
        len(summary_text),
        keep_turns,
    )

    return system_msgs + [compacted_msg] + tail


# -- Orchestrator -------------------------------------------------------------


async def compact_if_needed(
    messages: list[dict],
    client,
    thresholds: CompactionThresholds | None = None,
) -> tuple[list[dict], str | None]:
    """Apply compaction if history tokens exceed thresholds.

    Returns (compacted_messages, action_taken).
    action_taken is None | 'snip' | 'summary' | 'emergency'.
    """
    t = thresholds or _default_thresholds()
    tokens = estimate_tokens(messages)

    if tokens < t.soft_limit:
        return messages, None

    if tokens < t.hard_limit:
        result = snip_compact(messages, t.keep_turns)
        return result, "snip"

    # Hard limit exceeded — use summary
    result = await summary_compact(
        messages,
        client,
        keep_turns=t.keep_turns,
        max_summary_tokens=400,
        max_bullets=t.max_bullets,
    )

    # Emergency: if still over emergency_limit, drop oldest messages
    post_tokens = estimate_tokens(result)
    if post_tokens <= t.emergency_limit:
        return result, "summary"

    # Emergency drop: remove oldest user+assistant messages from preserved tail
    # Never drop system or [COMPACTED] messages
    emergency_result = list(result)
    while estimate_tokens(emergency_result) > t.emergency_limit:
        # Find first user/assistant message after system/compacted header
        drop_idx = None
        for i, msg in enumerate(emergency_result):
            if msg.get("role") in ("user", "assistant"):
                drop_idx = i
                break
        if drop_idx is None:
            break  # only system messages left, can't drop more
        emergency_result.pop(drop_idx)

    logger.warning(
        "emergency_compact: pre=%d post=%d limit=%d",
        post_tokens,
        estimate_tokens(emergency_result),
        t.emergency_limit,
    )

    return emergency_result, "emergency"
