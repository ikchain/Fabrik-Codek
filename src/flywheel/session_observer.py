"""Session Observer - Aprende de las conversaciones con Claude Code.

Este módulo procesa los transcripts de Claude Code y extrae
training pairs para que fabrik-codek aprenda de cómo Claude
resuelve problemas.

Uso:
    python -m src.flywheel.session_observer process
    python -m src.flywheel.session_observer watch  # Modo continuo
"""

import asyncio
import json
import logging
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from src.utils.hashing import hash_id
from src.utils.io import atomic_write_text

logger = logging.getLogger(__name__)

# Rutas de Claude Code
CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"
DATALAKE_PATH = Path(__file__).parent.parent.parent / "data"
SESSIONS_OUTPUT = DATALAKE_PATH / "01-raw" / "sessions"
TRAINING_OUTPUT = DATALAKE_PATH / "02-processed" / "training-pairs"
PROCESSED_MARKER = DATALAKE_PATH / "03-metadata" / ".processed_sessions"


def get_processed_sessions() -> set[str]:
    """Get list of already processed session files."""
    if not PROCESSED_MARKER.exists():
        return set()
    return set(PROCESSED_MARKER.read_text().strip().split("\n"))


def mark_session_processed(session_file: str) -> None:
    """Mark a session file as processed."""
    PROCESSED_MARKER.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_MARKER, "a") as f:
        f.write(f"{session_file}\n")


def extract_conversation_pairs(session_file: Path) -> Iterator[dict]:
    """Extract user/assistant pairs from a Claude Code session file."""
    messages = []

    with open(session_file, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get("type") in ("user", "assistant"):
                    msg = entry.get("message", {})
                    content = msg.get("content", "")

                    # Handle content that might be a list (tool calls, etc.)
                    if isinstance(content, list):
                        # Extract text parts
                        text_parts = []
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text_parts.append(part.get("text", ""))
                            elif isinstance(part, str):
                                text_parts.append(part)
                        content = "\n".join(text_parts)

                    if content:
                        messages.append(
                            {
                                "role": entry["type"],
                                "content": content,
                                "timestamp": entry.get("timestamp", ""),
                            }
                        )
            except json.JSONDecodeError:
                continue

    # Pair up user/assistant messages
    current_user = None
    for msg in messages:
        if msg["role"] == "user":
            current_user = msg
        elif msg["role"] == "assistant" and current_user:
            yield {
                "instruction": current_user["content"],
                "output": msg["content"],
                "timestamp": msg.get("timestamp", ""),
            }
            current_user = None


def categorize_interaction(instruction: str, output: str) -> str:
    """Categorize the type of interaction for better training."""
    instruction_lower = instruction.lower()

    # Code-related
    if any(
        kw in instruction_lower
        for kw in ["escribe", "write", "create", "implement", "función", "function", "class"]
    ):
        if "```" in output:
            return "code_generation"

    # Bug fixing
    if any(kw in instruction_lower for kw in ["fix", "error", "bug", "arregla", "corrige"]):
        return "error_fix"

    # Refactoring
    if any(
        kw in instruction_lower for kw in ["refactor", "improve", "optimize", "mejora", "optimiza"]
    ):
        return "refactor"

    # Explanation
    if any(
        kw in instruction_lower
        for kw in ["explain", "what", "how", "why", "explica", "qué", "cómo", "por qué"]
    ):
        return "explanation"

    # Review
    if any(kw in instruction_lower for kw in ["review", "check", "revisa", "mira"]):
        return "code_review"

    # Decision
    if any(
        kw in instruction_lower
        for kw in ["should", "better", "recommend", "debería", "mejor", "recomienda"]
    ):
        return "decision"

    return "general"


def calculate_quality_score(instruction: str, output: str) -> float:
    """Estimate quality score for training pair (0-1)."""
    score = 0.5  # Base score

    # Longer, detailed outputs are usually better
    if len(output) > 500:
        score += 0.1
    if len(output) > 1000:
        score += 0.1

    # Code in output is valuable
    if "```" in output:
        score += 0.15

    # Structured output (lists, headers) indicates quality
    if any(marker in output for marker in ["- ", "* ", "1.", "##", "**"]):
        score += 0.05

    # Very short outputs might be less valuable
    if len(output) < 100:
        score -= 0.2

    # Single word/yes/no responses are low quality
    if len(output.split()) < 10:
        score -= 0.3

    return max(0.0, min(1.0, score))


def process_session(session_file: Path, min_quality: float = 0.4) -> list[dict]:
    """Process a single session file and extract training pairs."""
    training_pairs = []

    for pair in extract_conversation_pairs(session_file):
        instruction = pair["instruction"]
        output = pair["output"]

        # Skip very short interactions
        if len(instruction) < 10 or len(output) < 20:
            continue

        quality = calculate_quality_score(instruction, output)
        if quality < min_quality:
            continue

        category = categorize_interaction(instruction, output)

        # Create training pair
        training_pair = {
            # FC-94: SHA256[:16] (64-bit). Legacy MD5 ids in older files are left
            # as-is — this output is append-only, old ids are never re-looked-up.
            "id": hash_id(f"{instruction[:100]}{output[:100]}"),
            "instruction": instruction,
            "input": "",  # Context could go here
            "output": output,
            "category": category,
            "quality_score": quality,
            "source": "claude-code-session",
            "source_file": session_file.name,
            "extracted_at": datetime.now().isoformat(),
        }

        training_pairs.append(training_pair)

    return training_pairs


def process_all_sessions(min_quality: float = 0.4) -> dict:
    """Process all unprocessed Claude Code sessions."""
    processed = get_processed_sessions()
    stats = {
        "sessions_processed": 0,
        "pairs_extracted": 0,
        "by_category": {},
    }

    # Ensure output directories exist
    SESSIONS_OUTPUT.mkdir(parents=True, exist_ok=True)
    TRAINING_OUTPUT.mkdir(parents=True, exist_ok=True)

    all_pairs = []

    # Find all session files
    for project_dir in CLAUDE_PROJECTS_DIR.iterdir():
        if not project_dir.is_dir():
            continue

        for session_file in project_dir.glob("*.jsonl"):
            # Include all files (sessions and agents have the same format)
            # Skip already processed
            file_key = f"{project_dir.name}/{session_file.name}"
            if file_key in processed:
                continue

            # Process session
            pairs = process_session(session_file, min_quality)

            if pairs:
                all_pairs.extend(pairs)
                stats["sessions_processed"] += 1
                stats["pairs_extracted"] += len(pairs)

                # Count by category
                for pair in pairs:
                    cat = pair["category"]
                    stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1

            # Mark as processed
            mark_session_processed(file_key)

    # Write all pairs to training file
    if all_pairs:
        output_file = (
            TRAINING_OUTPUT / f"claude-sessions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        lines = "".join(json.dumps(pair, ensure_ascii=False) + "\n" for pair in all_pairs)
        atomic_write_text(output_file, lines)

        stats["output_file"] = str(output_file)

    return stats


def get_stats() -> dict:
    """Get statistics about processed sessions."""
    processed = get_processed_sessions()

    # Count training pairs
    total_pairs = 0
    for jsonl_file in TRAINING_OUTPUT.glob("claude-sessions_*.jsonl"):
        with open(jsonl_file) as f:
            total_pairs += sum(1 for _ in f)

    return {
        "sessions_processed": len(processed),
        "total_training_pairs": total_pairs,
        "training_dir": str(TRAINING_OUTPUT),
    }


async def watch_sessions(
    interval_seconds: int = 60,
    min_quality: float = 0.4,
) -> None:
    """Continuously poll for new sessions and process them.

    Runs process_all_sessions() every interval_seconds. Logs when new
    sessions are found. Designed to be cancelled via asyncio.CancelledError
    or by the caller setting a stop event.
    """
    logger.info("watch_started, interval=%d", interval_seconds)

    while True:
        try:
            stats = process_all_sessions(min_quality=min_quality)
            if stats["sessions_processed"] > 0:
                logger.info(
                    "watch_cycle_processed, sessions=%d, pairs=%d",
                    stats["sessions_processed"],
                    stats["pairs_extracted"],
                )
        except Exception as e:
            logger.warning("watch_cycle_error, error=%s", str(e))
        await asyncio.sleep(interval_seconds)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.flywheel.session_observer [process|stats|reset]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "process":
        print("Processing Claude Code sessions...")
        stats = process_all_sessions()
        print("\nResults:")
        print(f"  Sessions processed: {stats['sessions_processed']}")
        print(f"  Training pairs extracted: {stats['pairs_extracted']}")
        if stats.get("by_category"):
            print("  By category:")
            for cat, count in sorted(stats["by_category"].items(), key=lambda x: -x[1]):
                print(f"    - {cat}: {count}")
        if stats.get("output_file"):
            print(f"  Output: {stats['output_file']}")

    elif command == "stats":
        stats = get_stats()
        print("Session Observer Stats:")
        print(f"  Sessions processed: {stats['sessions_processed']}")
        print(f"  Total training pairs: {stats['total_training_pairs']}")
        print(f"  Training dir: {stats['training_dir']}")

    elif command == "reset":
        if PROCESSED_MARKER.exists():
            PROCESSED_MARKER.unlink()
            print("Reset complete. All sessions will be reprocessed.")
        else:
            print("Nothing to reset.")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
