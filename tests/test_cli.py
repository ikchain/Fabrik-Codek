"""Tests for the Fabrik-Codek CLI (src/interfaces/cli.py).

Uses typer.testing.CliRunner to invoke commands and unittest.mock to
isolate every external dependency (LLM, graph, datalake, flywheel, etc.).

Patching strategy: CLI functions use lazy imports inside function bodies
(e.g. ``from src.core import LLMClient``).  Python resolves this through
the __init__.py re-export, so we patch at the package __init__ level
(e.g. ``src.core.LLMClient``, ``src.knowledge.DatalakeConnector``).
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from src.interfaces.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_response(**overrides):
    """Build a mock LLMResponse-like object."""
    resp = MagicMock()
    resp.content = overrides.get("content", "mock response")
    resp.model = overrides.get("model", "qwen2.5-coder:7b")
    resp.tokens_used = overrides.get("tokens_used", 42)
    resp.latency_ms = overrides.get("latency_ms", 100.0)
    return resp


def _make_async_context_manager(instance):
    """Wrap *instance* so it works as ``async with Cls() as obj``."""
    instance.__aenter__ = AsyncMock(return_value=instance)
    instance.__aexit__ = AsyncMock(return_value=False)
    return instance


# ===================================================================
# TestStatus
# ===================================================================

class TestStatus:
    """Tests for the ``status`` command."""

    def _run_status(self, *, datalake_exists=True, flywheel_enabled=True,
                    graph_loaded=True, ollama_ok=True):
        """Helper to invoke ``status`` with controlled mocks."""
        mock_settings = MagicMock()
        mock_settings.default_model = "qwen2.5-coder:7b"
        mock_settings.datalake_path = MagicMock()
        mock_settings.datalake_path.exists.return_value = datalake_exists
        mock_settings.datalake_path.__str__ = lambda self: "/data/datalake"
        mock_settings.flywheel_enabled = flywheel_enabled
        mock_settings.flywheel_batch_size = 100

        mock_engine = MagicMock()
        mock_engine.load.return_value = graph_loaded
        mock_engine.get_stats.return_value = {"entity_count": 50, "edge_count": 120}

        mock_client = MagicMock()
        mock_client.health_check = AsyncMock(return_value=ollama_ok)
        mock_client = _make_async_context_manager(mock_client)

        with (
            patch("src.config.settings", mock_settings),
            patch("src.config.Settings", return_value=mock_settings),
            patch("src.knowledge.GraphEngine", return_value=mock_engine),
            patch("src.knowledge.graph_engine.GraphEngine", return_value=mock_engine),
            patch("src.core.LLMClient", return_value=mock_client),
            patch("src.core.llm_client.LLMClient", return_value=mock_client),
        ):
            return runner.invoke(app, ["status"])

    def test_status_all_healthy(self):
        """Status shows all green when everything is available."""
        result = self._run_status()
        assert result.exit_code == 0
        assert "Fabrik-Codek Status" in result.output
        assert "Ollama conectado" in result.output

    def test_status_graph_not_built(self):
        """Status shows graph not built when load() returns False."""
        result = self._run_status(graph_loaded=False)
        assert result.exit_code == 0
        assert "No construido" in result.output

    def test_status_ollama_down(self):
        """Status reports Ollama unavailable."""
        result = self._run_status(ollama_ok=False)
        assert result.exit_code == 0
        assert "Ollama no disponible" in result.output

    def test_status_datalake_missing(self):
        """Status shows datalake missing when path does not exist."""
        result = self._run_status(datalake_exists=False)
        assert result.exit_code == 0
        assert "Fabrik-Codek Status" in result.output


# ===================================================================
# TestModels
# ===================================================================

class TestModels:
    """Tests for the ``models`` command."""

    def _patch_llm(self, mock_client):
        """Return a context manager that patches LLMClient at all import paths."""
        return (
            patch("src.core.LLMClient", return_value=mock_client),
        )

    def test_models_lists_available(self):
        """models command lists models from Ollama."""
        mock_client = MagicMock()
        mock_client.health_check = AsyncMock(return_value=True)
        mock_client.list_models = AsyncMock(return_value=["qwen2.5-coder:7b", "qwen2.5-coder:7b"])
        mock_client = _make_async_context_manager(mock_client)

        with patch("src.core.LLMClient", return_value=mock_client):
            result = runner.invoke(app, ["models"])

        assert result.exit_code == 0
        assert "qwen2.5-coder:7b" in result.output
        assert "qwen2.5-coder:7b" in result.output

    def test_models_ollama_down(self):
        """models command shows error when Ollama is not available."""
        mock_client = MagicMock()
        mock_client.health_check = AsyncMock(return_value=False)
        mock_client = _make_async_context_manager(mock_client)

        with patch("src.core.LLMClient", return_value=mock_client):
            result = runner.invoke(app, ["models"])

        assert result.exit_code == 0
        assert "Ollama no disponible" in result.output

    def test_models_empty_list(self):
        """models command handles empty model list gracefully."""
        mock_client = MagicMock()
        mock_client.health_check = AsyncMock(return_value=True)
        mock_client.list_models = AsyncMock(return_value=[])
        mock_client = _make_async_context_manager(mock_client)

        with patch("src.core.LLMClient", return_value=mock_client):
            result = runner.invoke(app, ["models"])

        assert result.exit_code == 0
        assert "Modelo" in result.output


# ===================================================================
# TestGraph
# ===================================================================

class TestGraph:
    """Tests for the ``graph`` command group."""

    def test_graph_stats_with_data(self):
        """graph stats shows statistics when graph is loaded."""
        mock_engine = MagicMock()
        mock_engine.load.return_value = True
        mock_engine.get_stats.return_value = {
            "entity_count": 150,
            "edge_count": 300,
            "connected_components": 5,
            "graph_path": "/tmp/graph.json",
            "entity_types": {"technology": 80, "concept": 70},
            "relation_types": {"uses": 200, "depends_on": 100},
        }

        with patch("src.knowledge.graph_engine.GraphEngine", return_value=mock_engine):
            result = runner.invoke(app, ["graph", "stats"])

        assert result.exit_code == 0
        assert "Knowledge Graph Stats" in result.output
        assert "150" in result.output
        assert "300" in result.output

    def test_graph_stats_no_graph(self):
        """graph stats shows message when no graph is built."""
        mock_engine = MagicMock()
        mock_engine.load.return_value = False

        with patch("src.knowledge.graph_engine.GraphEngine", return_value=mock_engine):
            result = runner.invoke(app, ["graph", "stats"])

        assert result.exit_code == 0
        assert "No hay Knowledge Graph" in result.output

    def test_graph_build_success(self):
        """graph build runs pipeline and shows results."""
        mock_engine = MagicMock()
        mock_engine.get_stats.return_value = {"entity_count": 100, "edge_count": 200}

        mock_pipeline = MagicMock()
        mock_pipeline.build = AsyncMock(return_value={
            "files_processed": 10,
            "pairs_processed": 500,
            "triples_extracted": 200,
            "errors": 0,
        })

        with (
            patch("src.knowledge.graph_engine.GraphEngine", return_value=mock_engine),
            patch("src.knowledge.extraction.pipeline.ExtractionPipeline", return_value=mock_pipeline),
        ):
            result = runner.invoke(app, ["graph", "build"])

        assert result.exit_code == 0
        assert "Knowledge Graph Build Complete" in result.output
        assert "10" in result.output

    def test_graph_search_finds_entities(self):
        """graph search shows matching entities."""
        mock_entity = MagicMock()
        mock_entity.name = "FastAPI"
        mock_entity.entity_type = MagicMock()
        mock_entity.entity_type.value = "technology"
        mock_entity.mention_count = 15
        mock_entity.aliases = ["fastapi"]
        mock_entity.id = "tech_fastapi"

        mock_neighbor = MagicMock()
        mock_neighbor.name = "Python"
        mock_neighbor.id = "tech_python"

        mock_relation = MagicMock()
        mock_relation.target_id = "tech_python"
        mock_relation.relation_type = MagicMock()
        mock_relation.relation_type.value = "uses"

        mock_engine = MagicMock()
        mock_engine.load.return_value = True
        mock_engine.search_entities.return_value = [mock_entity]
        mock_engine.get_neighbors.return_value = [(mock_neighbor, 0.9)]
        mock_engine.get_relations.return_value = [mock_relation]

        with patch("src.knowledge.graph_engine.GraphEngine", return_value=mock_engine):
            result = runner.invoke(app, ["graph", "search", "-q", "FastAPI"])

        assert result.exit_code == 0
        assert "FastAPI" in result.output
        assert "technology" in result.output

    def test_graph_search_no_results(self):
        """graph search shows message when no entities found."""
        mock_engine = MagicMock()
        mock_engine.load.return_value = True
        mock_engine.search_entities.return_value = []

        with patch("src.knowledge.graph_engine.GraphEngine", return_value=mock_engine):
            result = runner.invoke(app, ["graph", "search", "-q", "nonexistent"])

        assert result.exit_code == 0
        assert "No se encontraron" in result.output

    def test_graph_complete_success(self):
        """graph complete runs inference and saves."""
        mock_engine = MagicMock()
        mock_engine.load.return_value = True
        mock_engine.complete.return_value = {
            "inferred_count": 5,
            "depends_on_inferred": 3,
            "part_of_inferred": 2,
        }

        with patch("src.knowledge.graph_engine.GraphEngine", return_value=mock_engine):
            result = runner.invoke(app, ["graph", "complete"])

        assert result.exit_code == 0
        assert "Graph Completion Done" in result.output
        assert "5" in result.output
        mock_engine.save.assert_called_once()

    def test_graph_complete_no_graph(self):
        """graph complete shows error when no graph built."""
        mock_engine = MagicMock()
        mock_engine.load.return_value = False

        with patch("src.knowledge.graph_engine.GraphEngine", return_value=mock_engine):
            result = runner.invoke(app, ["graph", "complete"])

        assert result.exit_code == 0
        assert "No hay Knowledge Graph" in result.output

    def test_graph_unknown_action(self):
        """graph with unknown action shows usage hint."""
        mock_engine = MagicMock()

        with patch("src.knowledge.graph_engine.GraphEngine", return_value=mock_engine):
            result = runner.invoke(app, ["graph", "unknown_action"])

        assert result.exit_code == 0
        assert "graph build" in result.output or "graph search" in result.output


# ===================================================================
# TestLearn
# ===================================================================

class TestLearn:
    """Tests for the ``learn`` command."""

    def test_learn_process_success(self):
        """learn process runs session observer and shows stats."""
        mock_stats = {
            "sessions_processed": 5,
            "pairs_extracted": 120,
            "by_category": {"debugging": 40, "code_review": 80},
            "output_file": "/tmp/output.jsonl",
        }

        with patch(
            "src.flywheel.session_observer.process_all_sessions",
            return_value=mock_stats,
        ):
            result = runner.invoke(app, ["learn", "process"])

        assert result.exit_code == 0
        assert "Session Observer" in result.output
        assert "5" in result.output
        assert "120" in result.output

    def test_learn_process_shows_categories(self):
        """learn process displays per-category breakdown."""
        mock_stats = {
            "sessions_processed": 2,
            "pairs_extracted": 30,
            "by_category": {"debugging": 20, "testing": 10},
        }

        with patch(
            "src.flywheel.session_observer.process_all_sessions",
            return_value=mock_stats,
        ):
            result = runner.invoke(app, ["learn", "process"])

        assert result.exit_code == 0
        assert "debugging" in result.output

    def test_learn_stats(self):
        """learn stats shows learning statistics."""
        mock_stats = {
            "sessions_processed": 10,
            "total_training_pairs": 500,
        }

        with patch(
            "src.flywheel.session_observer.get_stats",
            return_value=mock_stats,
        ):
            result = runner.invoke(app, ["learn", "stats"])

        assert result.exit_code == 0
        assert "Learning Stats" in result.output
        assert "10" in result.output
        assert "500" in result.output

    def test_learn_reset_with_marker(self):
        """learn reset removes processed marker."""
        mock_marker = MagicMock()
        mock_marker.exists.return_value = True

        with patch(
            "src.flywheel.session_observer.PROCESSED_MARKER",
            mock_marker,
        ):
            result = runner.invoke(app, ["learn", "reset"])

        assert result.exit_code == 0
        assert "Reset completo" in result.output
        mock_marker.unlink.assert_called_once()

    def test_learn_reset_nothing_to_reset(self):
        """learn reset when no marker exists."""
        mock_marker = MagicMock()
        mock_marker.exists.return_value = False

        with patch(
            "src.flywheel.session_observer.PROCESSED_MARKER",
            mock_marker,
        ):
            result = runner.invoke(app, ["learn", "reset"])

        assert result.exit_code == 0
        assert "Nada que resetear" in result.output


# ===================================================================
# TestRag
# ===================================================================

class TestRag:
    """Tests for the ``rag`` command."""

    def test_rag_stats(self):
        """rag stats shows RAG engine statistics."""
        mock_engine = MagicMock()
        mock_engine.get_stats.return_value = {
            "total_documents": 1200,
            "db_path": "/tmp/lancedb",
        }
        mock_engine = _make_async_context_manager(mock_engine)

        with patch("src.knowledge.rag.RAGEngine", return_value=mock_engine):
            result = runner.invoke(app, ["rag", "stats"])

        assert result.exit_code == 0
        assert "RAG Stats" in result.output
        assert "1200" in result.output

    def test_rag_index(self):
        """rag index shows indexing results."""
        mock_engine = MagicMock()
        mock_engine.index_datalake = AsyncMock(return_value={
            "files_indexed": 50,
            "chunks_created": 300,
            "errors": 2,
        })
        mock_engine = _make_async_context_manager(mock_engine)

        with patch("src.knowledge.rag.RAGEngine", return_value=mock_engine):
            result = runner.invoke(app, ["rag", "index"])

        assert result.exit_code == 0
        assert "RAG Indexing Complete" in result.output
        assert "50" in result.output
        assert "300" in result.output

    def test_rag_search_with_results(self):
        """rag search -q shows matching documents."""
        mock_engine = MagicMock()
        mock_engine.retrieve = AsyncMock(return_value=[
            {"category": "debugging", "score": 0.951, "text": "Fix timeout error in API " * 20},
            {"category": "testing", "score": 0.852, "text": "Unit test for login flow " * 20},
        ])
        mock_engine = _make_async_context_manager(mock_engine)

        with patch("src.knowledge.rag.RAGEngine", return_value=mock_engine):
            result = runner.invoke(app, ["rag", "search", "-q", "timeout"])

        assert result.exit_code == 0
        assert "Resultados para" in result.output
        assert "0.951" in result.output
        assert "Fix timeout error" in result.output

    def test_rag_unknown_action(self):
        """rag with unknown action shows usage hint."""
        mock_engine = MagicMock()
        mock_engine = _make_async_context_manager(mock_engine)

        with patch("src.knowledge.rag.RAGEngine", return_value=mock_engine):
            result = runner.invoke(app, ["rag", "badaction"])

        assert result.exit_code == 0
        assert "rag index" in result.output or "rag search" in result.output


# ===================================================================
# TestDatalake
# ===================================================================

class TestDatalake:
    """Tests for the ``datalake`` command."""

    def test_datalake_stats(self):
        """datalake stats shows datalake statistics."""
        mock_connector = MagicMock()
        mock_connector.get_stats = AsyncMock(return_value={
            "total_files": 800,
            "total_size_mb": 45.50,
            "by_datalake": {"fabrik-codek-datalake": 800},
            "by_category": {"training_data": 500, "code_change": 300},
        })

        with patch("src.knowledge.DatalakeConnector", return_value=mock_connector):
            result = runner.invoke(app, ["datalake", "stats"])

        assert result.exit_code == 0
        assert "800" in result.output
        assert "45.50" in result.output

    def test_datalake_search(self):
        """datalake search shows matching files."""
        mock_file = MagicMock()
        mock_file.datalake = "fabrik-codek-datalake"
        mock_file.relative_path = "02-processed/training-pairs/agents_01.jsonl"
        mock_file.category = "training_data"
        mock_file.size = 1024

        mock_connector = MagicMock()
        mock_connector.search_files = AsyncMock(return_value=[mock_file])

        with patch("src.knowledge.DatalakeConnector", return_value=mock_connector):
            result = runner.invoke(app, ["datalake", "search", "-q", "agents"])

        assert result.exit_code == 0
        assert "agents" in result.output
        assert "1 encontrado" in result.output

    def test_datalake_decisions(self):
        """datalake decisions shows technical decisions."""
        mock_file = MagicMock()
        mock_file.relative_path = "03-metadata/decisions/decision_001.jsonl"

        mock_connector = MagicMock()
        mock_connector.get_decisions = AsyncMock(return_value=[mock_file])

        with patch("src.knowledge.DatalakeConnector", return_value=mock_connector):
            result = runner.invoke(app, ["datalake", "decisions"])

        assert result.exit_code == 0
        assert "Decisiones" in result.output

    def test_datalake_learnings(self):
        """datalake learnings shows learning entries."""
        mock_file = MagicMock()
        mock_file.relative_path = "03-metadata/learnings/learning_001.jsonl"

        mock_connector = MagicMock()
        mock_connector.get_learnings = AsyncMock(return_value=[mock_file])

        with patch("src.knowledge.DatalakeConnector", return_value=mock_connector):
            result = runner.invoke(app, ["datalake", "learnings"])

        assert result.exit_code == 0
        assert "Learnings" in result.output

    def test_datalake_unknown_action(self):
        """datalake with bad action shows usage hint."""
        mock_connector = MagicMock()

        with patch("src.knowledge.DatalakeConnector", return_value=mock_connector):
            result = runner.invoke(app, ["datalake", "badaction"])

        assert result.exit_code == 0
        assert "stats" in result.output


# ===================================================================
# TestFlywheel
# ===================================================================

class TestFlywheel:
    """Tests for the ``flywheel`` command."""

    def test_flywheel_stats(self):
        """flywheel stats shows flywheel status."""
        mock_collector = MagicMock()
        mock_collector.get_session_stats = AsyncMock(return_value={
            "enabled": True,
            "session_id": "abc12345-6789-0000-1111-222233334444",
            "buffered_records": 3,
        })

        with patch("src.flywheel.get_collector", return_value=mock_collector):
            result = runner.invoke(app, ["flywheel", "stats"])

        assert result.exit_code == 0
        assert "Flywheel Status" in result.output
        assert "abc12345" in result.output
        assert "3" in result.output

    def test_flywheel_export(self):
        """flywheel export shows export path."""
        mock_collector = MagicMock()
        mock_collector.export_training_pairs = AsyncMock(
            return_value="/tmp/training_export.jsonl"
        )

        with patch("src.flywheel.get_collector", return_value=mock_collector):
            result = runner.invoke(app, ["flywheel", "export"])

        assert result.exit_code == 0
        assert "Exportado" in result.output or "training_export" in result.output

    def test_flywheel_flush(self):
        """flywheel flush clears the buffer."""
        mock_collector = MagicMock()
        mock_collector.flush = AsyncMock()

        with patch("src.flywheel.get_collector", return_value=mock_collector):
            result = runner.invoke(app, ["flywheel", "flush"])

        assert result.exit_code == 0
        assert "Buffer vaciado" in result.output


# ===================================================================
# TestAsk
# ===================================================================

class TestAsk:
    """Tests for the ``ask`` command."""

    def _run_ask(self, args, *, content="mock response", tokens_used=42,
                 latency_ms=100.0, model="qwen2.5-coder:7b"):
        """Helper to invoke ``ask`` with controlled mocks."""
        mock_resp = _make_llm_response(
            content=content, tokens_used=tokens_used,
            latency_ms=latency_ms, model=model,
        )

        mock_client = MagicMock()
        mock_client.generate = AsyncMock(return_value=mock_resp)
        mock_client = _make_async_context_manager(mock_client)

        mock_collector = MagicMock()
        mock_collector.capture_prompt_response = AsyncMock()
        mock_collector.close = AsyncMock()

        with (
            patch("src.core.LLMClient", return_value=mock_client),
            patch("src.flywheel.get_collector", return_value=mock_collector),
        ):
            result = runner.invoke(app, args)

        return result, mock_client, mock_collector

    def test_ask_basic(self):
        """ask with a simple prompt returns a response."""
        result, _, _ = self._run_ask(
            ["ask", "What is the meaning of life?"],
            content="The answer is 42",
        )
        assert result.exit_code == 0
        assert "42" in result.output

    def test_ask_shows_token_and_latency(self):
        """ask output includes token count and latency."""
        result, _, _ = self._run_ask(
            ["ask", "test prompt"],
            content="done", tokens_used=150, latency_ms=250.5,
        )
        assert result.exit_code == 0
        assert "150 tokens" in result.output
        assert "250" in result.output

    def test_ask_captures_to_flywheel(self):
        """ask captures prompt/response to flywheel collector."""
        result, _, mock_collector = self._run_ask(
            ["ask", "capture this"],
            content="captured",
        )
        assert result.exit_code == 0
        mock_collector.capture_prompt_response.assert_called_once()
        mock_collector.close.assert_called_once()

    def test_ask_with_model_option(self):
        """ask --model passes model to LLMClient."""
        mock_resp = _make_llm_response()

        mock_client = MagicMock()
        mock_client.generate = AsyncMock(return_value=mock_resp)
        mock_client = _make_async_context_manager(mock_client)

        mock_collector = MagicMock()
        mock_collector.capture_prompt_response = AsyncMock()
        mock_collector.close = AsyncMock()

        with (
            patch("src.core.LLMClient", return_value=mock_client) as cls,
            patch("src.flywheel.get_collector", return_value=mock_collector),
        ):
            result = runner.invoke(
                app, ["ask", "Hello", "--model", "qwen2.5-coder:7b"]
            )

        assert result.exit_code == 0
        cls.assert_called_once_with(model="qwen2.5-coder:7b")


# ===================================================================
# TestFinetune
# ===================================================================

class TestFinetune:
    """Tests for the ``finetune`` command."""

    def test_finetune_dry_run(self):
        """finetune --dry-run shows data stats without training."""
        mock_stats = {
            "sessions_processed": 20,
            "total_training_pairs": 8955,
        }

        with patch(
            "src.flywheel.session_observer.get_stats",
            return_value=mock_stats,
        ):
            result = runner.invoke(app, ["finetune", "--dry-run"])

        assert result.exit_code == 0
        assert "Fine-tuning Data" in result.output
        assert "8955" in result.output
        assert "sin --dry-run" in result.output

    def test_finetune_dry_run_with_options(self):
        """finetune --dry-run shows configured epochs and batch."""
        mock_stats = {
            "sessions_processed": 20,
            "total_training_pairs": 1000,
        }

        with patch(
            "src.flywheel.session_observer.get_stats",
            return_value=mock_stats,
        ):
            result = runner.invoke(
                app, ["finetune", "--dry-run", "--epochs", "5", "--batch-size", "4"]
            )

        assert result.exit_code == 0
        assert "5" in result.output
        assert "4" in result.output

    @patch("subprocess.run")
    def test_finetune_runs_training_script(self, mock_subprocess):
        """finetune without --dry-run calls the training subprocess."""
        result = runner.invoke(app, ["finetune", "--epochs", "2", "--batch-size", "4"])

        assert result.exit_code == 0
        mock_subprocess.assert_called_once()
        cmd = mock_subprocess.call_args[0][0]
        assert "finetune.py" in cmd[1]
        assert "--epochs" in cmd
        assert "2" in cmd

    @patch("subprocess.run")
    def test_finetune_with_max_samples(self, mock_subprocess):
        """finetune --max-samples passes limit to subprocess."""
        result = runner.invoke(
            app, ["finetune", "--epochs", "1", "--max-samples", "100"]
        )

        assert result.exit_code == 0
        cmd = mock_subprocess.call_args[0][0]
        assert "--max-samples" in cmd
        assert "100" in cmd


# ===================================================================
# TestInit
# ===================================================================

class TestInit:
    """Tests for the ``init`` command."""

    def test_init_help(self):
        """init --help shows initialization description."""
        result = runner.invoke(app, ["init", "--help"])
        assert result.exit_code == 0
        assert "Initialize" in result.output or "init" in result.output

    @patch("shutil.which", return_value="/usr/bin/ollama")
    @patch("urllib.request.urlopen")
    @patch("subprocess.run")
    def test_init_full_setup(self, mock_subproc, mock_urlopen, mock_which, tmp_path):
        """init runs full setup when Ollama is available."""
        # Mock Ollama running
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        # Mock ollama list showing models exist
        mock_subproc.return_value = MagicMock(
            stdout="qwen2.5-coder:7b\nnomic-embed-text\n",
            returncode=0,
        )

        # Use tmp dir as project root
        env_example = tmp_path / ".env.example"
        env_example.write_text("FABRIK_DEFAULT_MODEL=qwen2.5-coder:7b\n")

        with patch("src.interfaces.cli.Path") as mock_path_cls:
            # __file__ resolution chain: cli.py -> interfaces -> src -> project_root
            mock_file = tmp_path / "src" / "interfaces" / "cli.py"
            mock_path_cls.__file__ = str(mock_file)
            mock_path_cls.return_value = mock_path_cls

            # Simpler approach: patch at module level
            with patch.object(Path, '__new__', wraps=Path.__new__):
                result = runner.invoke(app, ["init", "--skip-models"])

        assert result.exit_code == 0
        assert "Fabrik-Codek Setup" in result.output
        assert "Ollama installed" in result.output

    @patch("shutil.which", return_value=None)
    def test_init_no_ollama(self, mock_which, tmp_path):
        """init warns when Ollama is not installed."""
        result = runner.invoke(app, ["init", "--skip-models"])
        assert result.exit_code == 0
        assert "Ollama not installed" in result.output

    @patch("shutil.which", return_value="/usr/bin/ollama")
    @patch("urllib.request.urlopen", side_effect=Exception("Connection refused"))
    def test_init_ollama_not_running(self, mock_urlopen, mock_which):
        """init warns when Ollama is installed but not running."""
        result = runner.invoke(app, ["init", "--skip-models"])
        assert result.exit_code == 0
        assert "Ollama installed" in result.output
        assert "not running" in result.output

    def test_init_skip_models_flag(self):
        """init --skip-models skips model download."""
        with patch("shutil.which", return_value="/usr/bin/ollama"):
            with patch("urllib.request.urlopen") as mock_url:
                mock_resp = MagicMock()
                mock_resp.status = 200
                mock_resp.__enter__ = MagicMock(return_value=mock_resp)
                mock_resp.__exit__ = MagicMock(return_value=False)
                mock_url.return_value = mock_resp

                result = runner.invoke(app, ["init", "--skip-models"])

        assert result.exit_code == 0
        assert "Setup Complete" in result.output


# ===================================================================
# TestMCP
# ===================================================================

class TestMCP:
    """Tests for the ``mcp`` command."""

    def test_mcp_help(self):
        """mcp --help shows MCP server description."""
        result = runner.invoke(app, ["mcp", "--help"])
        assert result.exit_code == 0
        assert "MCP" in result.output or "mcp" in result.output

    def test_mcp_transport_option(self):
        """mcp --help shows transport option."""
        result = runner.invoke(app, ["mcp", "--help"])
        assert result.exit_code == 0
        assert "transport" in result.output.lower() or "stdio" in result.output.lower()
