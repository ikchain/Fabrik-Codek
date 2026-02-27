# Fix mark_feedback Persistence — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `FlywheelCollector.mark_feedback()` persist feedback to disk via a sidecar JSONL file, and bridge it with OutcomeTracker so feedback flows automatically.

**Architecture:** Feedback sidecar file (`feedback/*.jsonl`) — append-only, merged at read time by `export_training_pairs()`. OutcomeTracker's `record_turn()` return value drives `mark_feedback()` calls in CLI chat loop.

**Tech Stack:** Python 3.11+, asyncio, aiofiles, structlog, pytest

---

### Task 1: FeedbackRecord dataclass + sidecar persistence

**Files:**
- Modify: `src/flywheel/collector.py`
- Test: `tests/test_flywheel.py`

**Step 1: Write the failing tests**

Add to `tests/test_flywheel.py`:

```python
import json


class TestFeedbackPersistence:
    """Tests for mark_feedback sidecar persistence."""

    @pytest.mark.asyncio
    async def test_mark_feedback_persists_to_sidecar(self, collector, temp_data_dir):
        """Feedback on flushed record writes to sidecar file."""
        # Capture enough to trigger flush (batch_size=5)
        records = []
        for i in range(5):
            r = await collector.capture_prompt_response(
                prompt=f"prompt {i}", response=f"response {i}",
            )
            records.append(r)

        # Buffer is flushed, record is on disk only
        stats = await collector.get_session_stats()
        assert stats["buffered_records"] == 0

        # Mark feedback on flushed record
        await collector.mark_feedback(records[0].id, "positive")

        # Verify sidecar file exists with correct content
        feedback_dir = temp_data_dir / "feedback"
        files = list(feedback_dir.glob("*.jsonl"))
        assert len(files) == 1

        content = files[0].read_text().strip()
        entry = json.loads(content)
        assert entry["record_id"] == records[0].id
        assert entry["feedback"] == "positive"
        assert entry["was_edited"] is False
        assert entry["source"] == "manual"

    @pytest.mark.asyncio
    async def test_mark_feedback_buffer_priority(self, collector, temp_data_dir):
        """Feedback on buffered record updates buffer, no sidecar write."""
        record = await collector.capture_prompt_response(
            prompt="test", response="response",
        )

        await collector.mark_feedback(record.id, "negative", was_edited=True)

        # Buffer updated
        for r in collector._buffer:
            if r.id == record.id:
                assert r.user_feedback == "negative"
                assert r.was_edited is True

        # No sidecar written
        feedback_dir = temp_data_dir / "feedback"
        files = list(feedback_dir.glob("*.jsonl"))
        assert len(files) == 0

    @pytest.mark.asyncio
    async def test_mark_feedback_source_field(self, collector, temp_data_dir):
        """Source field is persisted correctly."""
        for i in range(5):
            await collector.capture_prompt_response(
                prompt=f"p{i}", response=f"r{i}",
            )

        await collector.mark_feedback(
            "nonexistent-id", "positive", source="outcome_tracker",
        )

        feedback_dir = temp_data_dir / "feedback"
        files = list(feedback_dir.glob("*.jsonl"))
        assert len(files) == 1

        entry = json.loads(files[0].read_text().strip())
        assert entry["source"] == "outcome_tracker"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_flywheel.py::TestFeedbackPersistence -v`
Expected: FAIL — `mark_feedback()` does not accept `source` parameter, no sidecar file created.

**Step 3: Implement FeedbackRecord and update mark_feedback**

In `src/flywheel/collector.py`:

1. Add `FeedbackRecord` dataclass after `InteractionRecord`:

```python
@dataclass
class FeedbackRecord:
    """Sidecar record for feedback on flushed interactions."""

    record_id: str
    feedback: Literal["positive", "negative", "neutral"]
    was_edited: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "manual"  # "manual" | "outcome_tracker"
```

2. In `__init__`, add after line 107:

```python
(self.data_dir / "feedback").mkdir(exist_ok=True)
```

3. Replace `mark_feedback` entirely:

```python
async def mark_feedback(
    self,
    record_id: str,
    feedback: Literal["positive", "negative", "neutral"],
    was_edited: bool = False,
    source: str = "manual",
) -> None:
    """Mark feedback for a captured interaction.

    If the record is still in the in-memory buffer, updates it directly.
    Otherwise, appends a FeedbackRecord to the sidecar JSONL file so
    that feedback is never lost after a buffer flush.
    """
    # Try buffer first (fast path)
    async with self._lock:
        for record in self._buffer:
            if record.id == record_id:
                record.user_feedback = feedback
                record.was_edited = was_edited
                logger.info(
                    "flywheel_feedback_buffer",
                    record_id=record_id,
                    feedback=feedback,
                )
                return

    # Record already flushed — persist to sidecar
    await self._persist_feedback(
        FeedbackRecord(
            record_id=record_id,
            feedback=feedback,
            was_edited=was_edited,
            source=source,
        )
    )

async def _persist_feedback(self, fb: FeedbackRecord) -> None:
    """Append a FeedbackRecord to today's sidecar JSONL file."""
    today = datetime.now().strftime("%Y-%m-%d")
    filepath = self.data_dir / "feedback" / f"{today}_feedback.jsonl"
    try:
        async with aiofiles.open(filepath, "a", encoding="utf-8") as f:
            line = json.dumps(asdict(fb), ensure_ascii=False)
            await f.write(line + "\n")
        logger.info(
            "flywheel_feedback_sidecar",
            record_id=fb.record_id,
            feedback=fb.feedback,
            source=fb.source,
        )
    except OSError as exc:
        logger.warning(
            "flywheel_feedback_write_failed",
            record_id=fb.record_id,
            error=str(exc),
        )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_flywheel.py::TestFeedbackPersistence -v`
Expected: 3 PASS

**Step 5: Run full test suite to check no regressions**

Run: `pytest tests/test_flywheel.py -v`
Expected: All existing tests still pass (TestInteractionRecord + TestFlywheelCollector + TestFeedbackPersistence)

**Step 6: Commit**

```bash
git add src/flywheel/collector.py tests/test_flywheel.py
git commit -m "FEAT: Add FeedbackRecord sidecar persistence"
```

---

### Task 2: Feedback index loading + export_training_pairs merge

**Files:**
- Modify: `src/flywheel/collector.py`
- Test: `tests/test_flywheel.py`

**Step 1: Write the failing tests**

Add to `tests/test_flywheel.py`:

```python
class TestFeedbackMerge:
    """Tests for feedback merge in export_training_pairs."""

    @pytest.mark.asyncio
    async def test_export_merges_feedback(self, collector, temp_data_dir):
        """Exported training pairs include sidecar feedback."""
        # Capture and flush
        records = []
        for i in range(5):
            r = await collector.capture_prompt_response(
                prompt=f"prompt {i}", response=f"response {i}",
            )
            records.append(r)

        # Mark feedback on flushed record
        await collector.mark_feedback(records[2].id, "positive")

        # Export
        output = await collector.export_training_pairs()

        # Read exported pairs
        pairs = []
        async with aiofiles.open(output, "r", encoding="utf-8") as f:
            content = await f.read()
            for line in content.strip().split("\n"):
                if line:
                    pairs.append(json.loads(line))

        # Find the pair for records[2]
        feedbacked = [p for p in pairs if p["instruction"] == "prompt 2"]
        assert len(feedbacked) == 1
        assert feedbacked[0]["metadata"]["feedback"] == "positive"

    @pytest.mark.asyncio
    async def test_feedback_last_wins(self, collector, temp_data_dir):
        """Multiple feedbacks for same record — most recent wins."""
        for i in range(5):
            await collector.capture_prompt_response(
                prompt=f"p{i}", response=f"r{i}",
            )

        target_id = collector._buffer[0].id if collector._buffer else "dummy"

        # Force flush to ensure buffer empty, feedback goes to sidecar
        await collector.flush()

        await collector.mark_feedback(target_id, "negative")
        await collector.mark_feedback(target_id, "positive")

        index = collector._load_feedback_index()
        assert index[target_id].feedback == "positive"

    @pytest.mark.asyncio
    async def test_load_feedback_index_empty(self, collector, temp_data_dir):
        """No feedback files returns empty dict without errors."""
        index = collector._load_feedback_index()
        assert index == {}
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_flywheel.py::TestFeedbackMerge -v`
Expected: FAIL — `_load_feedback_index` does not exist.

**Step 3: Implement _load_feedback_index and update export_training_pairs**

In `src/flywheel/collector.py`:

1. Add `_load_feedback_index` method to `FlywheelCollector`:

```python
def _load_feedback_index(self) -> dict[str, FeedbackRecord]:
    """Load all sidecar feedback files into a lookup dict.

    When multiple entries exist for the same record_id, the one with
    the latest timestamp wins.
    """
    index: dict[str, FeedbackRecord] = {}
    feedback_dir = self.data_dir / "feedback"

    if not feedback_dir.exists():
        return index

    for filepath in sorted(feedback_dir.glob("*.jsonl")):
        try:
            with filepath.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        fb = FeedbackRecord(**data)
                        existing = index.get(fb.record_id)
                        if existing is None or fb.timestamp > existing.timestamp:
                            index[fb.record_id] = fb
                    except (json.JSONDecodeError, TypeError) as exc:
                        logger.warning(
                            "feedback_parse_error",
                            file=str(filepath),
                            error=str(exc),
                        )
        except OSError as exc:
            logger.warning(
                "feedback_read_error",
                file=str(filepath),
                error=str(exc),
            )

    return index
```

2. Modify `export_training_pairs` — add feedback merge after loading records.
   Insert at the start of the method (after `training_pairs = []`):

```python
feedback_index = self._load_feedback_index()
```

   In the inner loop, after `record = InteractionRecord(**record_data)`, add:

```python
# Merge sidecar feedback if available
fb = feedback_index.get(record.id)
if fb is not None:
    record.user_feedback = fb.feedback
    record.was_edited = fb.was_edited
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_flywheel.py::TestFeedbackMerge -v`
Expected: 3 PASS

**Step 5: Run full test suite**

Run: `pytest tests/test_flywheel.py -v`
Expected: All pass

**Step 6: Commit**

```bash
git add src/flywheel/collector.py tests/test_flywheel.py
git commit -m "FEAT: Add feedback index merge in export_training_pairs"
```

---

### Task 3: OutcomeTracker bridge in CLI chat

**Files:**
- Modify: `src/interfaces/cli.py`
- Test: `tests/test_cli.py`

**Step 1: Write the failing test**

First, check what test infrastructure exists for CLI.

Look at `tests/test_cli.py` for existing patterns — use same mocking approach.

Add test to `tests/test_cli.py`:

```python
class TestOutcomeFeedbackBridge:
    """Tests for OutcomeTracker → mark_feedback bridge."""

    @pytest.mark.asyncio
    async def test_outcome_accepted_triggers_positive_feedback(self):
        """When OutcomeTracker infers 'accepted', mark_feedback is called
        with 'positive'."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from src.flywheel.collector import FlywheelCollector
        from src.flywheel.outcome_tracker import OutcomeTracker, OutcomeRecord

        collector = MagicMock(spec=FlywheelCollector)
        collector.mark_feedback = AsyncMock()

        outcome = OutcomeRecord(
            query="how do I do X",
            response_summary="do Y",
            task_type="explanation",
            model="test",
            outcome="accepted",
            inference_reason="topic change",
        )

        # Import the bridge function
        from src.flywheel.collector import bridge_outcome_to_feedback

        await bridge_outcome_to_feedback(collector, outcome, "record-123")

        collector.mark_feedback.assert_called_once_with(
            "record-123", "positive", source="outcome_tracker",
        )

    @pytest.mark.asyncio
    async def test_outcome_rejected_triggers_negative_feedback(self):
        """When OutcomeTracker infers 'rejected', mark_feedback is called
        with 'negative'."""
        from unittest.mock import AsyncMock, MagicMock
        from src.flywheel.collector import FlywheelCollector, bridge_outcome_to_feedback
        from src.flywheel.outcome_tracker import OutcomeRecord

        collector = MagicMock(spec=FlywheelCollector)
        collector.mark_feedback = AsyncMock()

        outcome = OutcomeRecord(
            query="how do I do X",
            response_summary="do Y",
            task_type="explanation",
            model="test",
            outcome="rejected",
            inference_reason="reformulation",
        )

        await bridge_outcome_to_feedback(collector, outcome, "record-456")

        collector.mark_feedback.assert_called_once_with(
            "record-456", "negative", source="outcome_tracker",
        )

    @pytest.mark.asyncio
    async def test_outcome_neutral_no_feedback(self):
        """Neutral outcome does not trigger mark_feedback."""
        from unittest.mock import AsyncMock, MagicMock
        from src.flywheel.collector import FlywheelCollector, bridge_outcome_to_feedback
        from src.flywheel.outcome_tracker import OutcomeRecord

        collector = MagicMock(spec=FlywheelCollector)
        collector.mark_feedback = AsyncMock()

        outcome = OutcomeRecord(
            query="how do I do X",
            response_summary="do Y",
            task_type="explanation",
            model="test",
            outcome="neutral",
            inference_reason="session_close",
        )

        await bridge_outcome_to_feedback(collector, outcome, "record-789")

        collector.mark_feedback.assert_not_called()

    @pytest.mark.asyncio
    async def test_outcome_none_no_feedback(self):
        """None outcome (first turn) does not trigger mark_feedback."""
        from unittest.mock import AsyncMock, MagicMock
        from src.flywheel.collector import FlywheelCollector, bridge_outcome_to_feedback

        collector = MagicMock(spec=FlywheelCollector)
        collector.mark_feedback = AsyncMock()

        await bridge_outcome_to_feedback(collector, None, "record-000")

        collector.mark_feedback.assert_not_called()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py::TestOutcomeFeedbackBridge -v`
Expected: FAIL — `bridge_outcome_to_feedback` does not exist.

**Step 3: Implement bridge function and integrate in CLI**

In `src/flywheel/collector.py`, add after the `FlywheelCollector` class (module-level function):

```python
_OUTCOME_TO_FEEDBACK: dict[str, str] = {
    "accepted": "positive",
    "rejected": "negative",
}


async def bridge_outcome_to_feedback(
    collector: FlywheelCollector,
    outcome: Any,
    record_id: str,
) -> None:
    """Bridge OutcomeTracker inference to FlywheelCollector feedback.

    Maps accepted → positive, rejected → negative. Neutral and None
    are ignored (no useful signal).
    """
    if outcome is None:
        return

    feedback = _OUTCOME_TO_FEEDBACK.get(getattr(outcome, "outcome", ""), "")
    if not feedback:
        return

    await collector.mark_feedback(record_id, feedback, source="outcome_tracker")
```

In `src/interfaces/cli.py`, modify the `chat()` function's inner loop.

After the `capture_prompt_response` call (around line 117-124), track the record id.
After `tracker.record_turn()` (around line 127-132), use the return value:

```python
# Capture for flywheel
record = await collector.capture_prompt_response(
    prompt=user_input,
    response=response.content,
    model=response.model,
    tokens=response.tokens_used,
    latency_ms=response.latency_ms,
    interaction_type="prompt_response",
)

# Track outcome — record_turn returns the PREVIOUS turn's outcome
prev_outcome = tracker.record_turn(
    query=user_input,
    response=response.content,
    decision=initial_decision,
    latency_ms=response.latency_ms,
)

# Bridge: outcome → feedback on the PREVIOUS record
if prev_outcome is not None and last_record_id is not None:
    from src.flywheel.collector import bridge_outcome_to_feedback
    await bridge_outcome_to_feedback(collector, prev_outcome, last_record_id)

last_record_id = record.id
```

Initialize `last_record_id = None` before the `while True` loop.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py::TestOutcomeFeedbackBridge -v`
Expected: 4 PASS

**Step 5: Run full test suite**

Run: `pytest tests/ -v --timeout=30`
Expected: All 783+ tests pass

**Step 6: Commit**

```bash
git add src/flywheel/collector.py src/interfaces/cli.py tests/test_cli.py
git commit -m "FEAT: Bridge OutcomeTracker to mark_feedback in CLI"
```

---

### Task 4: Final verification and cleanup

**Files:**
- All modified files

**Step 1: Run complete test suite**

Run: `pytest tests/ -v --timeout=30 2>&1 | tail -20`
Expected: All tests pass, no warnings related to new code.

**Step 2: Manual smoke test**

```bash
# Verify feedback directory gets created
fabrik status

# Quick chat to trigger the bridge
fabrik ask "what is python"

# Check if feedback sidecar was written (if outcomes were inferred)
ls -la "$(python -c 'from src.config import settings; print(settings.datalake_path)')/01-raw/feedback/"
```

**Step 3: Update CLAUDE.md**

Add under `## Data Flywheel` section, after the hook information:

```markdown
### Feedback Sidecar
- `mark_feedback()` persists to `01-raw/feedback/{YYYY-MM-DD}_feedback.jsonl`
- OutcomeTracker bridge: accepted → positive, rejected → negative (zero-friction)
- `export_training_pairs()` merges feedback at read time (last-write-wins)
```

**Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "DOCS: Update CLAUDE.md with feedback sidecar docs"
```

**Step 5: Transition Jira ticket to Done**

Mark  as Done in Jira.
