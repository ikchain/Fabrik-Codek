"""Data Flywheel Collector - Captures interactions for continuous learning.

This is the heart of the flywheel: while you work with Claude Code,
fabrik-codek learns from every interaction to improve over time.
"""

import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

import aiofiles
import structlog

from src.config import settings

logger = structlog.get_logger()


@dataclass
class InteractionRecord:
    """A single interaction record for the flywheel."""

    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Interaction data
    interaction_type: Literal[
        "prompt_response",  # User prompt -> LLM response
        "code_generation",  # Code generation task
        "code_review",  # Code review/feedback
        "refactor",  # Refactoring task
        "search",  # Search query
        "decision",  # Technical decision
        "error_fix",  # Error resolution
        "documentation",  # Doc generation
        "test",  # Test generation/execution
        "other",
    ] = "other"

    # Input/Output
    input_text: str = ""
    output_text: str = ""
    context: str = ""  # Additional context (file content, etc.)

    # Metadata
    model: str = ""
    tokens_used: int = 0
    latency_ms: float = 0

    # Quality signals (for future training)
    user_feedback: Literal["positive", "negative", "neutral", "none"] = "none"
    was_edited: bool = False  # User edited the output
    was_accepted: bool = True  # Output was used as-is

    # Source tracking
    source: str = "fabrik-codek"  # Which system generated this
    session_id: str = ""
    project_context: str = ""  # Project/repo being worked on

    def to_training_pair(self) -> dict:
        """Convert to training pair format."""
        return {
            "instruction": self.input_text,
            "input": self.context,
            "output": self.output_text,
            "metadata": {
                "type": self.interaction_type,
                "feedback": self.user_feedback,
                "accepted": self.was_accepted,
                "model": self.model,
            },
        }


class FlywheelCollector:
    """Collects and stores interaction data for continuous learning.

    The flywheel pattern:
    1. Capture: Every interaction is logged
    2. Store: Data is persisted in structured format
    3. Process: Periodic processing prepares training data
    4. Learn: Model fine-tuning improves over time
    5. Repeat: Better model → better interactions → better data
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        batch_size: int | None = None,
        enabled: bool | None = None,
    ):
        self.data_dir = data_dir or settings.raw_data_path
        self.batch_size = batch_size or settings.flywheel_batch_size
        self.enabled = enabled if enabled is not None else settings.flywheel_enabled

        self._buffer: list[InteractionRecord] = []
        self._session_id = str(uuid4())
        self._lock = asyncio.Lock()

        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "interactions").mkdir(exist_ok=True)
        (self.data_dir / "training_pairs").mkdir(exist_ok=True)

    async def capture(self, record: InteractionRecord) -> None:
        """Capture an interaction record."""
        if not self.enabled:
            return

        record.session_id = self._session_id

        async with self._lock:
            self._buffer.append(record)

            if len(self._buffer) >= self.batch_size:
                await self._flush()

        logger.debug(
            "flywheel_capture",
            type=record.interaction_type,
            tokens=record.tokens_used,
        )

    async def capture_prompt_response(
        self,
        prompt: str,
        response: str,
        model: str = "",
        tokens: int = 0,
        latency_ms: float = 0,
        context: str = "",
        interaction_type: str = "prompt_response",
    ) -> InteractionRecord:
        """Convenience method to capture a prompt-response pair."""
        record = InteractionRecord(
            interaction_type=interaction_type,
            input_text=prompt,
            output_text=response,
            context=context,
            model=model,
            tokens_used=tokens,
            latency_ms=latency_ms,
        )
        await self.capture(record)
        return record

    async def mark_feedback(
        self,
        record_id: str,
        feedback: Literal["positive", "negative", "neutral"],
        was_edited: bool = False,
    ) -> None:
        """Mark feedback for a captured interaction."""
        # Update in buffer if still there
        async with self._lock:
            for record in self._buffer:
                if record.id == record_id:
                    record.user_feedback = feedback
                    record.was_edited = was_edited
                    return

        # TODO: Update in persisted storage
        logger.info("flywheel_feedback", record_id=record_id, feedback=feedback)

    async def _flush(self) -> None:
        """Flush buffer to disk."""
        if not self._buffer:
            return

        today = datetime.now().strftime("%Y-%m-%d")
        filename = f"interactions_{today}_{self._session_id[:8]}.jsonl"
        filepath = self.data_dir / "interactions" / filename

        async with aiofiles.open(filepath, "a", encoding="utf-8") as f:
            for record in self._buffer:
                line = json.dumps(asdict(record), ensure_ascii=False)
                await f.write(line + "\n")

        count = len(self._buffer)
        self._buffer.clear()

        logger.info("flywheel_flush", count=count, file=filename)

    async def flush(self) -> None:
        """Public flush method."""
        async with self._lock:
            await self._flush()

    async def close(self) -> None:
        """Close collector and flush remaining data."""
        await self.flush()
        logger.info("flywheel_closed", session_id=self._session_id)

    async def get_session_stats(self) -> dict:
        """Get statistics for current session."""
        return {
            "session_id": self._session_id,
            "buffered_records": len(self._buffer),
            "enabled": self.enabled,
        }

    async def export_training_pairs(
        self,
        output_path: Path | None = None,
        min_feedback: Literal["positive", "neutral", "any"] = "any",
    ) -> Path:
        """Export captured data as training pairs.

        This processes raw interactions into instruction-tuning format.
        """
        output_path = output_path or (
            self.data_dir
            / "training_pairs"
            / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )

        interactions_dir = self.data_dir / "interactions"
        training_pairs = []

        # Process all interaction files
        for jsonl_file in interactions_dir.glob("*.jsonl"):
            async with aiofiles.open(jsonl_file, "r", encoding="utf-8") as f:
                content = await f.read()
                for line in content.strip().split("\n"):
                    if not line:
                        continue
                    try:
                        record_data = json.loads(line)
                        record = InteractionRecord(**record_data)

                        # Filter by feedback if specified
                        if min_feedback == "positive" and record.user_feedback != "positive":
                            continue
                        if min_feedback == "neutral" and record.user_feedback == "negative":
                            continue

                        # Skip if marked as not accepted
                        if not record.was_accepted:
                            continue

                        training_pairs.append(record.to_training_pair())
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning("invalid_record", error=str(e))

        # Write training pairs
        async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
            for pair in training_pairs:
                line = json.dumps(pair, ensure_ascii=False)
                await f.write(line + "\n")

        logger.info(
            "training_pairs_exported",
            count=len(training_pairs),
            path=str(output_path),
        )

        return output_path


# Global collector instance
_collector: FlywheelCollector | None = None


def get_collector() -> FlywheelCollector:
    """Get or create the global flywheel collector."""
    global _collector
    if _collector is None:
        _collector = FlywheelCollector()
    return _collector
