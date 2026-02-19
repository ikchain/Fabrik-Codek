"""Tests for the RAG engine."""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.knowledge.rag import RAGEngine, EMBEDDING_DIM


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _fake_embedding(dim=EMBEDDING_DIM):
    """Return a fake embedding vector."""
    return [0.1] * dim


def _make_engine():
    """Create a RAGEngine instance bypassing __init__ defaults."""
    engine = RAGEngine.__new__(RAGEngine)
    engine._db = None
    engine._table = None
    engine._http_client = None
    engine.embedding_model = "nomic-embed-text"
    engine.ollama_host = "http://localhost:11434"
    engine.table_name = "knowledge"
    return engine


def _setup_mock_http(engine):
    """Attach a mock HTTP client that returns fake embeddings."""
    engine._http_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": _fake_embedding()}
    mock_response.raise_for_status = MagicMock()
    engine._http_client.post = AsyncMock(return_value=mock_response)


class TestChunkText:
    """Tests for _chunk_text - pure logic, no mocking needed."""

    def test_short_text_no_chunking(self):
        engine = _make_engine()
        text = "Short text under chunk size."
        chunks = engine._chunk_text(text, chunk_size=1000)
        assert chunks == [text]

    def test_long_text_produces_chunks(self):
        engine = _make_engine()
        text = "A" * 2500
        chunks = engine._chunk_text(text, chunk_size=1000, overlap=200)
        assert len(chunks) >= 2

    def test_sentence_boundary_break(self):
        engine = _make_engine()
        # Create text where a period exists after half the chunk_size
        text = "A" * 600 + ". " + "B" * 600
        chunks = engine._chunk_text(text, chunk_size=1000, overlap=200)
        assert len(chunks) >= 2
        # First chunk should end near the sentence boundary
        assert chunks[0].endswith(".")

    def test_overlap_between_chunks(self):
        engine = _make_engine()
        text = "word " * 500  # ~2500 chars
        chunks = engine._chunk_text(text, chunk_size=1000, overlap=200)
        assert len(chunks) >= 2
        # Verify overlap: end of chunk N should appear at start of chunk N+1
        # (they share some characters due to overlap)
        for i in range(len(chunks) - 1):
            end_of_current = chunks[i][-100:]
            # Some portion of end should appear in next chunk start
            start_of_next = chunks[i + 1][:300]
            # With 200 char overlap there should be shared content
            assert len(set(end_of_current.split()) & set(start_of_next.split())) > 0

    def test_empty_string(self):
        engine = _make_engine()
        chunks = engine._chunk_text("", chunk_size=1000)
        assert chunks == [""]

    def test_exact_chunk_size(self):
        engine = _make_engine()
        text = "A" * 1000
        chunks = engine._chunk_text(text, chunk_size=1000)
        assert chunks == [text]

    def test_one_char_over_chunk_size(self):
        engine = _make_engine()
        text = "A" * 1001
        chunks = engine._chunk_text(text, chunk_size=1000, overlap=200)
        assert len(chunks) >= 2

    def test_newline_boundary_break(self):
        engine = _make_engine()
        # Create text where a newline exists after half the chunk_size
        text = "A" * 600 + "\n" + "B" * 600
        chunks = engine._chunk_text(text, chunk_size=1000, overlap=200)
        assert len(chunks) >= 2

    def test_no_boundary_in_first_half(self):
        engine = _make_engine()
        # Period only in the first quarter - should not be used as break point
        # because break_point must be > chunk_size // 2
        text = "A" * 100 + ". " + "B" * 1100
        chunks = engine._chunk_text(text, chunk_size=1000, overlap=200)
        assert len(chunks) >= 2

    def test_whitespace_stripped_from_chunks(self):
        engine = _make_engine()
        text = "  Hello world. " + "X" * 1000
        chunks = engine._chunk_text(text, chunk_size=1000, overlap=200)
        for chunk in chunks:
            assert chunk == chunk.strip()


class TestRAGEngineInit:
    """Tests for RAG engine initialization and lifecycle."""

    def test_default_settings(self):
        with patch("src.knowledge.rag.settings") as mock_settings:
            mock_settings.data_dir = Path("/tmp/test")
            mock_settings.embedding_model = "nomic-embed-text"
            mock_settings.ollama_host = "http://localhost:11434"
            engine = RAGEngine()
            assert engine.embedding_model == "nomic-embed-text"
            assert engine.ollama_host == "http://localhost:11434"
            assert engine.db_path == Path("/tmp/test/vectordb")
            assert engine.table_name == "knowledge"

    def test_custom_overrides(self):
        with patch("src.knowledge.rag.settings") as mock_settings:
            mock_settings.data_dir = Path("/tmp/fallback")
            mock_settings.embedding_model = "fallback"
            mock_settings.ollama_host = "http://fallback:1234"
            engine = RAGEngine(
                db_path=Path("/custom/path"),
                embedding_model="custom-model",
                ollama_host="http://custom:1234",
            )
            assert engine.db_path == Path("/custom/path")
            assert engine.embedding_model == "custom-model"
            assert engine.ollama_host == "http://custom:1234"

    def test_partial_overrides(self):
        with patch("src.knowledge.rag.settings") as mock_settings:
            mock_settings.data_dir = Path("/tmp/test")
            mock_settings.embedding_model = "default-model"
            mock_settings.ollama_host = "http://default:11434"
            engine = RAGEngine(embedding_model="custom-model")
            assert engine.embedding_model == "custom-model"
            assert engine.ollama_host == "http://default:11434"
            assert engine.db_path == Path("/tmp/test/vectordb")

    def test_context_manager_creates_table(self, tmp_dir):
        def _test():
            async def _run():
                with patch("src.knowledge.rag.lancedb") as mock_lance:
                    mock_db = MagicMock()
                    mock_db.table_names.return_value = []
                    mock_table = MagicMock()
                    mock_db.create_table.return_value = mock_table
                    mock_lance.connect.return_value = mock_db

                    engine = RAGEngine(db_path=tmp_dir / "test_rag")
                    async with engine as e:
                        assert e._http_client is not None
                        assert e._table is not None
                        assert e._table is mock_table
                    # After exit, client should be closed
                    assert e._http_client.is_closed
            asyncio.run(_run())
        _test()

    def test_init_opens_existing_table(self, tmp_dir):
        def _test():
            async def _run():
                with patch("src.knowledge.rag.lancedb") as mock_lance:
                    mock_db = MagicMock()
                    mock_db.table_names.return_value = ["knowledge"]
                    mock_table = MagicMock()
                    mock_db.open_table.return_value = mock_table
                    mock_lance.connect.return_value = mock_db

                    engine = RAGEngine(db_path=tmp_dir / "test_rag")
                    await engine._init()
                    mock_db.open_table.assert_called_once_with("knowledge")
                    mock_db.create_table.assert_not_called()
                    await engine.close()
            asyncio.run(_run())
        _test()

    def test_init_creates_new_table(self, tmp_dir):
        def _test():
            async def _run():
                with patch("src.knowledge.rag.lancedb") as mock_lance:
                    mock_db = MagicMock()
                    mock_db.table_names.return_value = []
                    mock_table = MagicMock()
                    mock_db.create_table.return_value = mock_table
                    mock_lance.connect.return_value = mock_db

                    engine = RAGEngine(db_path=tmp_dir / "test_rag")
                    await engine._init()
                    mock_db.create_table.assert_called_once()
                    mock_db.open_table.assert_not_called()
                    await engine.close()
            asyncio.run(_run())
        _test()

    def test_close_with_no_client(self):
        def _test():
            async def _run():
                engine = _make_engine()
                engine._http_client = None
                # Should not raise
                await engine.close()
            asyncio.run(_run())
        _test()

    def test_close_calls_aclose(self):
        def _test():
            async def _run():
                engine = _make_engine()
                mock_client = AsyncMock()
                engine._http_client = mock_client
                await engine.close()
                mock_client.aclose.assert_awaited_once()
            asyncio.run(_run())
        _test()


class TestEmbeddings:
    """Tests for embedding generation."""

    def test_get_single_embedding(self):
        def _test():
            async def _run():
                engine = _make_engine()
                _setup_mock_http(engine)

                result = await engine._get_embedding("test text")
                assert len(result) == EMBEDDING_DIM
                engine._http_client.post.assert_called_once()
                # Verify correct URL and payload
                call_args = engine._http_client.post.call_args
                assert "/api/embeddings" in call_args[0][0]
                assert call_args[1]["json"]["model"] == "nomic-embed-text"
                assert call_args[1]["json"]["prompt"] == "test text"
            asyncio.run(_run())
        _test()

    def test_get_embedding_raises_on_http_error(self):
        def _test():
            async def _run():
                engine = _make_engine()
                engine._http_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.raise_for_status.side_effect = Exception("HTTP 500")
                engine._http_client.post = AsyncMock(return_value=mock_response)

                with pytest.raises(Exception, match="HTTP 500"):
                    await engine._get_embedding("test text")
            asyncio.run(_run())
        _test()

    def test_batch_embeddings(self):
        def _test():
            async def _run():
                engine = _make_engine()
                _setup_mock_http(engine)

                texts = [f"text {i}" for i in range(25)]
                result = await engine._get_embeddings_batch(texts, batch_size=10)
                assert len(result) == 25
                assert engine._http_client.post.call_count == 25
                # Each result should be a valid embedding
                for emb in result:
                    assert len(emb) == EMBEDDING_DIM
            asyncio.run(_run())
        _test()

    def test_batch_embeddings_single_batch(self):
        def _test():
            async def _run():
                engine = _make_engine()
                _setup_mock_http(engine)

                texts = ["text 1", "text 2", "text 3"]
                result = await engine._get_embeddings_batch(texts, batch_size=10)
                assert len(result) == 3
                assert engine._http_client.post.call_count == 3
            asyncio.run(_run())
        _test()

    def test_batch_embeddings_empty(self):
        def _test():
            async def _run():
                engine = _make_engine()
                _setup_mock_http(engine)

                result = await engine._get_embeddings_batch([], batch_size=10)
                assert result == []
                engine._http_client.post.assert_not_called()
            asyncio.run(_run())
        _test()

    def test_batch_embeddings_exact_batch_size(self):
        def _test():
            async def _run():
                engine = _make_engine()
                _setup_mock_http(engine)

                texts = [f"text {i}" for i in range(10)]
                result = await engine._get_embeddings_batch(texts, batch_size=10)
                assert len(result) == 10
            asyncio.run(_run())
        _test()


class TestIndexFile:
    """Tests for file indexing."""

    def test_index_nonexistent_file(self):
        def _test():
            async def _run():
                engine = _make_engine()
                result = await engine.index_file(Path("/nonexistent/file.txt"))
                assert result == 0
            asyncio.run(_run())
        _test()

    def test_index_empty_file(self, tmp_dir):
        def _test():
            async def _run():
                engine = _make_engine()
                empty_file = tmp_dir / "empty.md"
                empty_file.write_text("")
                result = await engine.index_file(empty_file)
                assert result == 0
            asyncio.run(_run())
        _test()

    def test_index_whitespace_only_file(self, tmp_dir):
        def _test():
            async def _run():
                engine = _make_engine()
                ws_file = tmp_dir / "whitespace.md"
                ws_file.write_text("   \n\t\n  ")
                result = await engine.index_file(ws_file)
                assert result == 0
            asyncio.run(_run())
        _test()

    def test_index_valid_file(self, tmp_dir):
        def _test():
            async def _run():
                engine = _make_engine()
                _setup_mock_http(engine)
                engine._table = MagicMock()

                test_file = tmp_dir / "test.md"
                test_file.write_text("This is a test document with enough content to be indexed properly.")

                result = await engine.index_file(test_file, category="test", project="testproj")
                assert result > 0
                engine._table.add.assert_called_once()
                # Verify the documents passed to add
                added_docs = engine._table.add.call_args[0][0]
                assert len(added_docs) == result
                for doc in added_docs:
                    assert doc["category"] == "test"
                    assert doc["project"] == "testproj"
                    assert doc["source"] == str(test_file)
                    assert len(doc["vector"]) == EMBEDDING_DIM
            asyncio.run(_run())
        _test()

    def test_index_large_file_multiple_chunks(self, tmp_dir):
        def _test():
            async def _run():
                engine = _make_engine()
                _setup_mock_http(engine)
                engine._table = MagicMock()

                test_file = tmp_dir / "large.md"
                test_file.write_text("This is a sentence. " * 200)  # ~4000 chars

                result = await engine.index_file(test_file, category="docs")
                assert result > 1  # Should produce multiple chunks
                engine._table.add.assert_called_once()
            asyncio.run(_run())
        _test()

    def test_index_file_default_category_and_project(self, tmp_dir):
        def _test():
            async def _run():
                engine = _make_engine()
                _setup_mock_http(engine)
                engine._table = MagicMock()

                test_file = tmp_dir / "defaults.md"
                test_file.write_text("Some content for testing defaults in index_file method.")

                result = await engine.index_file(test_file)
                assert result > 0
                added_docs = engine._table.add.call_args[0][0]
                assert added_docs[0]["category"] == "general"
                assert added_docs[0]["project"] == ""
            asyncio.run(_run())
        _test()

    def test_index_file_unique_ids(self, tmp_dir):
        def _test():
            async def _run():
                engine = _make_engine()
                _setup_mock_http(engine)
                engine._table = MagicMock()

                test_file = tmp_dir / "multichunk.md"
                test_file.write_text("Word " * 500)  # Multiple chunks

                result = await engine.index_file(test_file)
                added_docs = engine._table.add.call_args[0][0]
                ids = [doc["id"] for doc in added_docs]
                # All IDs should be unique
                assert len(ids) == len(set(ids))
            asyncio.run(_run())
        _test()


class TestIndexJsonl:
    """Tests for JSONL indexing."""

    def test_index_nonexistent_jsonl(self):
        def _test():
            async def _run():
                engine = _make_engine()
                result = await engine.index_jsonl(Path("/nonexistent/file.jsonl"))
                assert result == 0
            asyncio.run(_run())
        _test()

    def test_index_valid_jsonl(self, tmp_dir):
        def _test():
            async def _run():
                engine = _make_engine()
                _setup_mock_http(engine)
                engine._table = MagicMock()

                jsonl_file = tmp_dir / "pairs.jsonl"
                pairs = [
                    {"instruction": "How to use FastAPI?", "output": "FastAPI is a modern web framework for building APIs. " * 5, "category": "api"},
                    {"instruction": "Explain Docker", "output": "Docker is a containerization platform that allows packaging. " * 5, "category": "devops"},
                ]
                with open(jsonl_file, "w") as f:
                    for p in pairs:
                        f.write(json.dumps(p) + "\n")

                result = await engine.index_jsonl(jsonl_file)
                assert result == 2
                engine._table.add.assert_called_once()
                added_docs = engine._table.add.call_args[0][0]
                assert len(added_docs) == 2
                # Text should combine instruction + output
                assert added_docs[0]["text"].startswith("Q: How to use FastAPI?")
                assert "A: FastAPI" in added_docs[0]["text"]
            asyncio.run(_run())
        _test()

    def test_index_jsonl_skips_short_text(self, tmp_dir):
        def _test():
            async def _run():
                engine = _make_engine()
                _setup_mock_http(engine)
                engine._table = MagicMock()

                jsonl_file = tmp_dir / "short.jsonl"
                with open(jsonl_file, "w") as f:
                    f.write(json.dumps({"instruction": "hi", "output": "bye"}) + "\n")

                result = await engine.index_jsonl(jsonl_file)
                assert result == 0
                engine._table.add.assert_not_called()
            asyncio.run(_run())
        _test()

    def test_index_jsonl_skips_malformed(self, tmp_dir):
        def _test():
            async def _run():
                engine = _make_engine()
                _setup_mock_http(engine)
                engine._table = MagicMock()

                jsonl_file = tmp_dir / "mixed.jsonl"
                with open(jsonl_file, "w") as f:
                    f.write("not valid json\n")
                    f.write(json.dumps({"instruction": "Q", "output": "A valid output with enough text to pass filter. " * 5}) + "\n")

                result = await engine.index_jsonl(jsonl_file)
                assert result == 1
            asyncio.run(_run())
        _test()

    def test_index_jsonl_custom_text_field(self, tmp_dir):
        def _test():
            async def _run():
                engine = _make_engine()
                _setup_mock_http(engine)
                engine._table = MagicMock()

                jsonl_file = tmp_dir / "custom.jsonl"
                with open(jsonl_file, "w") as f:
                    f.write(json.dumps({"description": "A detailed description of the code change that was made. " * 3}) + "\n")

                result = await engine.index_jsonl(jsonl_file, text_field="description")
                assert result == 1
            asyncio.run(_run())
        _test()

    def test_index_jsonl_no_instruction_field(self, tmp_dir):
        def _test():
            async def _run():
                engine = _make_engine()
                _setup_mock_http(engine)
                engine._table = MagicMock()

                jsonl_file = tmp_dir / "noinstruction.jsonl"
                with open(jsonl_file, "w") as f:
                    f.write(json.dumps({"output": "A long output without any instruction field to combine with. " * 3}) + "\n")

                result = await engine.index_jsonl(jsonl_file)
                assert result == 1
                added_docs = engine._table.add.call_args[0][0]
                # Without instruction, text should just be the output (no "Q:" prefix)
                assert not added_docs[0]["text"].startswith("Q:")
            asyncio.run(_run())
        _test()

    def test_index_jsonl_custom_category(self, tmp_dir):
        def _test():
            async def _run():
                engine = _make_engine()
                _setup_mock_http(engine)
                engine._table = MagicMock()

                jsonl_file = tmp_dir / "cat.jsonl"
                with open(jsonl_file, "w") as f:
                    f.write(json.dumps({"output": "Content with enough text to be indexed and pass the filter. " * 3}) + "\n")

                result = await engine.index_jsonl(jsonl_file, category="code_change")
                assert result == 1
                added_docs = engine._table.add.call_args[0][0]
                # Default category from data should be "code_change" (from param)
                assert added_docs[0]["category"] == "code_change"
            asyncio.run(_run())
        _test()

    def test_index_jsonl_uses_category_from_data(self, tmp_dir):
        def _test():
            async def _run():
                engine = _make_engine()
                _setup_mock_http(engine)
                engine._table = MagicMock()

                jsonl_file = tmp_dir / "cat_in_data.jsonl"
                with open(jsonl_file, "w") as f:
                    f.write(json.dumps({
                        "output": "Content with enough text to pass filter and be indexed properly. " * 3,
                        "category": "debugging"
                    }) + "\n")

                result = await engine.index_jsonl(jsonl_file, category="training")
                assert result == 1
                added_docs = engine._table.add.call_args[0][0]
                # Category from data should override parameter
                assert added_docs[0]["category"] == "debugging"
            asyncio.run(_run())
        _test()

    def test_index_jsonl_empty_file(self, tmp_dir):
        def _test():
            async def _run():
                engine = _make_engine()
                engine._table = MagicMock()

                jsonl_file = tmp_dir / "empty.jsonl"
                jsonl_file.write_text("")

                result = await engine.index_jsonl(jsonl_file)
                assert result == 0
                engine._table.add.assert_not_called()
            asyncio.run(_run())
        _test()

    def test_index_jsonl_project_from_source_file(self, tmp_dir):
        def _test():
            async def _run():
                engine = _make_engine()
                _setup_mock_http(engine)
                engine._table = MagicMock()

                jsonl_file = tmp_dir / "proj.jsonl"
                with open(jsonl_file, "w") as f:
                    f.write(json.dumps({
                        "output": "Content with enough text to pass the fifty character filter easily. " * 3,
                        "source_file": "myapp/src/main.py"
                    }) + "\n")

                result = await engine.index_jsonl(jsonl_file)
                assert result == 1
                added_docs = engine._table.add.call_args[0][0]
                assert added_docs[0]["project"] == "myapp"
            asyncio.run(_run())
        _test()


class TestIndexDatalake:
    """Tests for datalake indexing."""

    def test_missing_datalake(self, tmp_dir):
        def _test():
            async def _run():
                with patch("src.knowledge.rag.settings") as mock_settings:
                    mock_settings.datalake_path = tmp_dir / "nonexistent"
                    engine = _make_engine()
                    stats = await engine.index_datalake()
                    assert stats == {"files_indexed": 0, "chunks_created": 0, "errors": 0}
            asyncio.run(_run())
        _test()

    def test_datalake_indexes_training_pairs(self, tmp_dir):
        def _test():
            async def _run():
                # Set up mock datalake structure
                training_dir = tmp_dir / "02-processed" / "training-pairs"
                training_dir.mkdir(parents=True)
                jsonl_file = training_dir / "test.jsonl"
                jsonl_file.write_text(json.dumps({"instruction": "Q", "output": "A" * 100}) + "\n")

                with patch("src.knowledge.rag.settings") as mock_settings:
                    mock_settings.datalake_path = tmp_dir

                    engine = _make_engine()
                    engine.index_jsonl = AsyncMock(return_value=5)
                    engine.index_file = AsyncMock(return_value=1)

                    stats = await engine.index_datalake()
                    assert stats["files_indexed"] >= 1
                    assert stats["chunks_created"] >= 5
                    assert stats["errors"] == 0
                    engine.index_jsonl.assert_called()
            asyncio.run(_run())
        _test()

    def test_datalake_indexes_decisions(self, tmp_dir):
        def _test():
            async def _run():
                decisions_dir = tmp_dir / "03-metadata" / "decisions"
                decisions_dir.mkdir(parents=True)
                md_file = decisions_dir / "decision-001.md"
                md_file.write_text("# Decision: Use LanceDB\nWe chose LanceDB for vector storage.")

                with patch("src.knowledge.rag.settings") as mock_settings:
                    mock_settings.datalake_path = tmp_dir

                    engine = _make_engine()
                    engine.index_jsonl = AsyncMock(return_value=0)
                    engine.index_file = AsyncMock(return_value=2)

                    stats = await engine.index_datalake()
                    assert stats["files_indexed"] >= 1
                    # index_file should have been called with category="decision"
                    found_decision_call = False
                    for call in engine.index_file.call_args_list:
                        if call[1].get("category") == "decision" or (len(call[0]) > 1 and call[0][1] == "decision"):
                            found_decision_call = True
                    assert found_decision_call
            asyncio.run(_run())
        _test()

    def test_datalake_indexes_learnings(self, tmp_dir):
        def _test():
            async def _run():
                learnings_dir = tmp_dir / "03-metadata" / "learnings"
                learnings_dir.mkdir(parents=True)
                md_file = learnings_dir / "learning-001.md"
                md_file.write_text("# Learning: Always validate inputs")

                with patch("src.knowledge.rag.settings") as mock_settings:
                    mock_settings.datalake_path = tmp_dir

                    engine = _make_engine()
                    engine.index_jsonl = AsyncMock(return_value=0)
                    engine.index_file = AsyncMock(return_value=1)

                    stats = await engine.index_datalake()
                    assert stats["files_indexed"] >= 1
                    found_learning_call = False
                    for call in engine.index_file.call_args_list:
                        if call[1].get("category") == "learning" or (len(call[0]) > 1 and call[0][1] == "learning"):
                            found_learning_call = True
                    assert found_learning_call
            asyncio.run(_run())
        _test()

    def test_datalake_indexes_code_changes(self, tmp_dir):
        def _test():
            async def _run():
                code_dir = tmp_dir / "01-raw" / "code-changes"
                code_dir.mkdir(parents=True)
                jsonl_file = code_dir / "2026-02-01_auto-captures.jsonl"
                jsonl_file.write_text(json.dumps({"description": "Changed authentication" * 10}) + "\n")

                with patch("src.knowledge.rag.settings") as mock_settings:
                    mock_settings.datalake_path = tmp_dir

                    engine = _make_engine()
                    engine.index_jsonl = AsyncMock(return_value=3)
                    engine.index_file = AsyncMock(return_value=0)

                    stats = await engine.index_datalake()
                    assert stats["files_indexed"] >= 1
                    # Should call index_jsonl with text_field="description"
                    found_code_change = False
                    for call in engine.index_jsonl.call_args_list:
                        kwargs = call[1]
                        if kwargs.get("text_field") == "description" and kwargs.get("category") == "code_change":
                            found_code_change = True
                    assert found_code_change
            asyncio.run(_run())
        _test()

    def test_datalake_handles_errors(self, tmp_dir):
        def _test():
            async def _run():
                training_dir = tmp_dir / "02-processed" / "training-pairs"
                training_dir.mkdir(parents=True)
                jsonl_file = training_dir / "bad.jsonl"
                jsonl_file.write_text("dummy")

                with patch("src.knowledge.rag.settings") as mock_settings:
                    mock_settings.datalake_path = tmp_dir

                    engine = _make_engine()
                    engine.index_jsonl = AsyncMock(side_effect=OSError("Index failed"))
                    engine.index_file = AsyncMock(return_value=0)

                    stats = await engine.index_datalake()
                    assert stats["errors"] >= 1
            asyncio.run(_run())
        _test()

    def test_datalake_empty_dirs(self, tmp_dir):
        def _test():
            async def _run():
                # Create structure but empty dirs
                (tmp_dir / "02-processed" / "training-pairs").mkdir(parents=True)
                (tmp_dir / "03-metadata" / "decisions").mkdir(parents=True)
                (tmp_dir / "03-metadata" / "learnings").mkdir(parents=True)
                (tmp_dir / "01-raw" / "code-changes").mkdir(parents=True)

                with patch("src.knowledge.rag.settings") as mock_settings:
                    mock_settings.datalake_path = tmp_dir

                    engine = _make_engine()
                    engine.index_jsonl = AsyncMock(return_value=0)
                    engine.index_file = AsyncMock(return_value=0)

                    stats = await engine.index_datalake()
                    assert stats["files_indexed"] == 0
                    assert stats["chunks_created"] == 0
                    assert stats["errors"] == 0
            asyncio.run(_run())
        _test()


class TestRetrieve:
    """Tests for retrieval."""

    def test_retrieve_returns_results(self):
        def _test():
            async def _run():
                engine = _make_engine()
                _setup_mock_http(engine)

                # Mock table search chain: search() -> limit() -> to_list()
                mock_to_list = MagicMock()
                mock_to_list.to_list.return_value = [
                    {"text": "result1", "source": "file1", "category": "test", "_distance": 0.1},
                    {"text": "result2", "source": "file2", "category": "test", "_distance": 0.2},
                ]
                mock_search = MagicMock()
                mock_search.limit.return_value = mock_to_list
                engine._table = MagicMock()
                engine._table.search.return_value = mock_search

                results = await engine.retrieve("test query", limit=5)
                assert len(results) == 2
                assert results[0]["text"] == "result1"
                assert results[0]["score"] == 0.1
                assert results[1]["text"] == "result2"
                assert results[1]["score"] == 0.2
                mock_search.limit.assert_called_once_with(5)
            asyncio.run(_run())
        _test()

    def test_retrieve_with_category_filter(self):
        def _test():
            async def _run():
                engine = _make_engine()
                _setup_mock_http(engine)

                # Mock chain: search() -> limit() -> where() -> to_list()
                mock_where = MagicMock()
                mock_where.to_list.return_value = []
                mock_limit = MagicMock()
                mock_limit.where.return_value = mock_where
                mock_search = MagicMock()
                mock_search.limit.return_value = mock_limit
                engine._table = MagicMock()
                engine._table.search.return_value = mock_search

                results = await engine.retrieve("test", category="api")
                mock_limit.where.assert_called_once_with("category = 'api'")
                assert results == []
            asyncio.run(_run())
        _test()

    def test_retrieve_empty_results(self):
        def _test():
            async def _run():
                engine = _make_engine()
                _setup_mock_http(engine)

                mock_to_list = MagicMock()
                mock_to_list.to_list.return_value = []
                mock_search = MagicMock()
                mock_search.limit.return_value = mock_to_list
                engine._table = MagicMock()
                engine._table.search.return_value = mock_search

                results = await engine.retrieve("nonexistent topic")
                assert results == []
            asyncio.run(_run())
        _test()

    def test_retrieve_default_limit(self):
        def _test():
            async def _run():
                engine = _make_engine()
                _setup_mock_http(engine)

                mock_to_list = MagicMock()
                mock_to_list.to_list.return_value = []
                mock_search = MagicMock()
                mock_search.limit.return_value = mock_to_list
                engine._table = MagicMock()
                engine._table.search.return_value = mock_search

                await engine.retrieve("query")
                mock_search.limit.assert_called_once_with(5)  # Default limit
            asyncio.run(_run())
        _test()

    def test_retrieve_result_format(self):
        def _test():
            async def _run():
                engine = _make_engine()
                _setup_mock_http(engine)

                mock_to_list = MagicMock()
                mock_to_list.to_list.return_value = [
                    {"text": "sample", "source": "path/to/file", "category": "docs", "_distance": 0.05},
                ]
                mock_search = MagicMock()
                mock_search.limit.return_value = mock_to_list
                engine._table = MagicMock()
                engine._table.search.return_value = mock_search

                results = await engine.retrieve("test")
                assert len(results) == 1
                result = results[0]
                assert set(result.keys()) == {"text", "source", "category", "score"}
                assert result["text"] == "sample"
                assert result["source"] == "path/to/file"
                assert result["category"] == "docs"
                assert result["score"] == 0.05
            asyncio.run(_run())
        _test()

    def test_retrieve_missing_distance_field(self):
        def _test():
            async def _run():
                engine = _make_engine()
                _setup_mock_http(engine)

                mock_to_list = MagicMock()
                mock_to_list.to_list.return_value = [
                    {"text": "sample", "source": "file", "category": "docs"},
                ]
                mock_search = MagicMock()
                mock_search.limit.return_value = mock_to_list
                engine._table = MagicMock()
                engine._table.search.return_value = mock_search

                results = await engine.retrieve("test")
                assert results[0]["score"] == 0  # Default when _distance missing
            asyncio.run(_run())
        _test()


class TestQueryWithContext:
    """Tests for query_with_context."""

    def test_no_results_returns_original_query(self):
        def _test():
            async def _run():
                engine = _make_engine()
                engine.retrieve = AsyncMock(return_value=[])
                result = await engine.query_with_context("test query")
                assert result == "test query"
            asyncio.run(_run())
        _test()

    def test_results_inject_context(self):
        def _test():
            async def _run():
                engine = _make_engine()
                engine.retrieve = AsyncMock(return_value=[
                    {"text": "Context about FastAPI", "category": "api", "score": 0.1, "source": "f"},
                ])
                result = await engine.query_with_context("How to use FastAPI?")
                assert "Contexto relevante" in result
                assert "Context about FastAPI" in result
                assert "How to use FastAPI?" in result
                assert "Pregunta:" in result
            asyncio.run(_run())
        _test()

    def test_multiple_results_joined(self):
        def _test():
            async def _run():
                engine = _make_engine()
                engine.retrieve = AsyncMock(return_value=[
                    {"text": "First context", "category": "api", "score": 0.1, "source": "f1"},
                    {"text": "Second context", "category": "docs", "score": 0.2, "source": "f2"},
                ])
                result = await engine.query_with_context("test query")
                assert "First context" in result
                assert "Second context" in result
                assert "---" in result  # Separator between contexts
            asyncio.run(_run())
        _test()

    def test_context_includes_category_prefix(self):
        def _test():
            async def _run():
                engine = _make_engine()
                engine.retrieve = AsyncMock(return_value=[
                    {"text": "Some text", "category": "debugging", "score": 0.1, "source": "f"},
                ])
                result = await engine.query_with_context("test")
                assert "[debugging]" in result
            asyncio.run(_run())
        _test()

    def test_long_text_truncated_in_context(self):
        def _test():
            async def _run():
                engine = _make_engine()
                long_text = "A" * 1000
                engine.retrieve = AsyncMock(return_value=[
                    {"text": long_text, "category": "test", "score": 0.1, "source": "f"},
                ])
                result = await engine.query_with_context("test")
                # Text should be truncated to 500 chars in context
                # Count the A's in the result - should be at most 500
                context_section = result.split("Pregunta:")[0]
                a_count = context_section.count("A")
                assert a_count == 500
            asyncio.run(_run())
        _test()

    def test_custom_limit(self):
        def _test():
            async def _run():
                engine = _make_engine()
                engine.retrieve = AsyncMock(return_value=[])
                await engine.query_with_context("test", limit=10)
                engine.retrieve.assert_called_once_with("test", limit=10)
            asyncio.run(_run())
        _test()

    def test_default_limit_is_three(self):
        def _test():
            async def _run():
                engine = _make_engine()
                engine.retrieve = AsyncMock(return_value=[])
                await engine.query_with_context("test")
                engine.retrieve.assert_called_once_with("test", limit=3)
            asyncio.run(_run())
        _test()


class TestGetStats:
    """Tests for get_stats."""

    def test_no_table(self):
        engine = _make_engine()
        engine._table = None
        stats = engine.get_stats()
        assert stats == {"total_documents": 0}

    def test_with_table(self):
        engine = _make_engine()
        engine._table = MagicMock()
        engine._table.count_rows.return_value = 42
        engine.db_path = Path("/test/path")
        stats = engine.get_stats()
        assert stats["total_documents"] == 42
        assert stats["db_path"] == "/test/path"

    def test_table_error(self):
        engine = _make_engine()
        engine._table = MagicMock()
        engine._table.count_rows.side_effect = OSError("DB error")
        stats = engine.get_stats()
        assert stats == {"total_documents": 0}

    def test_zero_rows(self):
        engine = _make_engine()
        engine._table = MagicMock()
        engine._table.count_rows.return_value = 0
        engine.db_path = Path("/test/path")
        stats = engine.get_stats()
        assert stats["total_documents"] == 0
        assert stats["db_path"] == "/test/path"

    def test_large_table(self):
        engine = _make_engine()
        engine._table = MagicMock()
        engine._table.count_rows.return_value = 100000
        engine.db_path = Path("/test/path")
        stats = engine.get_stats()
        assert stats["total_documents"] == 100000


class TestGetRagEngine:
    """Tests for the singleton get_rag_engine function."""

    def test_get_rag_engine_creates_singleton(self):
        def _test():
            async def _run():
                import src.knowledge.rag as rag_module
                # Reset singleton
                rag_module._rag_engine = None

                with patch("src.knowledge.rag.settings") as mock_settings:
                    mock_settings.data_dir = Path("/tmp/test_singleton")
                    mock_settings.embedding_model = "test-model"
                    mock_settings.ollama_host = "http://localhost:11434"
                    with patch("src.knowledge.rag.lancedb") as mock_lance:
                        mock_db = MagicMock()
                        mock_db.table_names.return_value = []
                        mock_db.create_table.return_value = MagicMock()
                        mock_lance.connect.return_value = mock_db

                        engine1 = await rag_module.get_rag_engine()
                        engine2 = await rag_module.get_rag_engine()
                        assert engine1 is engine2

                # Clean up singleton
                if rag_module._rag_engine and rag_module._rag_engine._http_client:
                    await rag_module._rag_engine.close()
                rag_module._rag_engine = None
            asyncio.run(_run())
        _test()
