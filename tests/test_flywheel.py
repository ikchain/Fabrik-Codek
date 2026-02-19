"""Tests for flywheel collector."""

import pytest
from pathlib import Path
import tempfile

from src.flywheel.collector import FlywheelCollector, InteractionRecord


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def collector(temp_data_dir):
    """Create collector with temp directory."""
    return FlywheelCollector(data_dir=temp_data_dir, batch_size=5, enabled=True)


class TestInteractionRecord:
    """Tests for InteractionRecord."""

    def test_create_record(self):
        """Test creating a record."""
        record = InteractionRecord(
            interaction_type="code_generation",
            input_text="Write a hello world",
            output_text="print('Hello, World!')",
            model="qwen2.5-coder:7b",
        )

        assert record.interaction_type == "code_generation"
        assert record.input_text == "Write a hello world"
        assert record.id is not None
        assert record.timestamp is not None

    def test_to_training_pair(self):
        """Test conversion to training pair."""
        record = InteractionRecord(
            interaction_type="code_generation",
            input_text="Write a hello world",
            output_text="print('Hello, World!')",
            context="Python file",
            user_feedback="positive",
        )

        pair = record.to_training_pair()

        assert pair["instruction"] == "Write a hello world"
        assert pair["input"] == "Python file"
        assert pair["output"] == "print('Hello, World!')"
        assert pair["metadata"]["feedback"] == "positive"


class TestFlywheelCollector:
    """Tests for FlywheelCollector."""

    @pytest.mark.asyncio
    async def test_capture_record(self, collector):
        """Test capturing a record."""
        record = InteractionRecord(
            interaction_type="prompt_response",
            input_text="test",
            output_text="response",
        )

        await collector.capture(record)
        stats = await collector.get_session_stats()

        assert stats["buffered_records"] == 1

    @pytest.mark.asyncio
    async def test_capture_prompt_response(self, collector):
        """Test convenience capture method."""
        record = await collector.capture_prompt_response(
            prompt="Hello",
            response="Hi there!",
            model="test-model",
            tokens=10,
        )

        assert record.input_text == "Hello"
        assert record.output_text == "Hi there!"
        assert record.model == "test-model"

    @pytest.mark.asyncio
    async def test_flush_on_batch(self, collector, temp_data_dir):
        """Test auto-flush when batch size reached."""
        # Capture more than batch size
        for i in range(6):
            await collector.capture_prompt_response(
                prompt=f"prompt {i}",
                response=f"response {i}",
            )

        # Should have flushed 5, keeping 1 in buffer
        stats = await collector.get_session_stats()
        assert stats["buffered_records"] == 1

        # Check file was created
        interactions_dir = temp_data_dir / "interactions"
        files = list(interactions_dir.glob("*.jsonl"))
        assert len(files) == 1

    @pytest.mark.asyncio
    async def test_disabled_collector(self, temp_data_dir):
        """Test that disabled collector doesn't capture."""
        collector = FlywheelCollector(
            data_dir=temp_data_dir,
            enabled=False,
        )

        await collector.capture_prompt_response(
            prompt="test",
            response="response",
        )

        stats = await collector.get_session_stats()
        assert stats["buffered_records"] == 0

    @pytest.mark.asyncio
    async def test_mark_feedback(self, collector):
        """Test marking feedback on record."""
        record = await collector.capture_prompt_response(
            prompt="test",
            response="response",
        )

        await collector.mark_feedback(record.id, "positive", was_edited=True)

        # Verify feedback was set
        for r in collector._buffer:
            if r.id == record.id:
                assert r.user_feedback == "positive"
                assert r.was_edited is True
