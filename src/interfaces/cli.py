"""Fabrik-Codek CLI - Your local AI dev assistant."""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.markdown import Markdown

app = typer.Typer(
    name="fabrik",
    help="Fabrik-Codek: Local AI dev assistant powered by Qwen",
    add_completion=False,
)
console = Console()


def async_run(coro):
    """Run async function."""
    return asyncio.run(coro)


@app.command()
def chat(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    system: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt"),
):
    """Start interactive chat with the assistant."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory

    from src.config import settings
    from src.core import LLMClient
    from src.flywheel import get_collector

    console.print(
        Panel.fit(
            "[bold blue]Fabrik-Codek[/bold blue] - Tu hermano pequeño está listo 🤖",
            subtitle=f"Model: {model or settings.default_model}",
        )
    )
    console.print("[dim]Escribe 'exit' o 'quit' para salir, 'clear' para limpiar[/dim]\n")

    history_file = settings.data_dir / ".chat_history"
    session = PromptSession(history=FileHistory(str(history_file)))
    collector = get_collector()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    async def run_chat():
        async with LLMClient(model=model) as client:
            # Check health
            if not await client.health_check():
                console.print("[red]Error: Ollama no está disponible[/red]")
                console.print("[dim]Ejecuta: ollama serve[/dim]")
                return

            while True:
                try:
                    user_input = session.prompt("\n[You] → ")
                except (KeyboardInterrupt, EOFError):
                    break

                if not user_input.strip():
                    continue

                if user_input.lower() in ("exit", "quit", "q"):
                    break

                if user_input.lower() == "clear":
                    console.clear()
                    messages.clear()
                    if system:
                        messages.append({"role": "system", "content": system})
                    continue

                messages.append({"role": "user", "content": user_input})

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True,
                ) as progress:
                    progress.add_task("Pensando...", total=None)
                    response = await client.chat(messages)

                messages.append({"role": "assistant", "content": response.content})

                console.print(f"\n[bold green][Fabrik][/bold green] ({response.latency_ms:.0f}ms)")
                console.print(Markdown(response.content))

                # Capture for flywheel
                await collector.capture_prompt_response(
                    prompt=user_input,
                    response=response.content,
                    model=response.model,
                    tokens=response.tokens_used,
                    latency_ms=response.latency_ms,
                    interaction_type="prompt_response",
                )

        await collector.close()

    async_run(run_chat())
    console.print("\n[dim]¡Hasta luego! 👋[/dim]")


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="Question or task"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    context_file: Optional[Path] = typer.Option(None, "--context", "-c", help="File for context"),
    use_rag: bool = typer.Option(False, "--rag", "-r", help="Use RAG for context"),
    use_graph: bool = typer.Option(False, "--graph", "-g", help="Use hybrid RAG (vector + graph)"),
    graph_depth: int = typer.Option(2, "--depth", "-d", help="Graph traversal depth"),
):
    """Ask a single question."""
    from src.core import LLMClient
    from src.flywheel import get_collector

    context = ""
    final_prompt = prompt

    if context_file and context_file.exists():
        context = context_file.read_text()
        final_prompt = f"Context:\n```\n{context}\n```\n\nQuestion: {prompt}"

    async def run():
        nonlocal final_prompt, context

        # Inject hybrid RAG context (vector + graph)
        if use_graph:
            from src.knowledge.hybrid_rag import HybridRAGEngine
            async with HybridRAGEngine() as hybrid:
                results = await hybrid.retrieve(prompt, limit=5, graph_depth=graph_depth)
                if results:
                    final_prompt = await hybrid.query_with_context(
                        prompt, limit=5, graph_depth=graph_depth,
                    )
                    context = final_prompt
                    origins = {r.get("origin", "?") for r in results}
                    console.print(
                        f"[dim]Hybrid RAG: {len(results)} docs "
                        f"(origins: {', '.join(origins)})[/dim]\n"
                    )

        # Inject RAG context if requested (vector only)
        elif use_rag:
            from src.knowledge.rag import RAGEngine
            async with RAGEngine() as rag:
                rag_results = await rag.retrieve(prompt, limit=3)
                if rag_results:
                    rag_context = "\n---\n".join([
                        f"[{r['category']}] {r['text'][:500]}"
                        for r in rag_results
                    ])
                    final_prompt = f"""Contexto de tu base de conocimiento:
{rag_context}

---
Pregunta: {prompt}

Responde usando el contexto cuando sea relevante."""
                    context = rag_context
                    console.print(f"[dim]RAG: {len(rag_results)} documentos relevantes encontrados[/dim]\n")

        async with LLMClient(model=model) as client:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("Procesando...", total=None)
                response = await client.generate(final_prompt)

            console.print(Markdown(response.content))
            console.print(f"\n[dim]({response.tokens_used} tokens, {response.latency_ms:.0f}ms)[/dim]")

            collector = get_collector()
            await collector.capture_prompt_response(
                prompt=prompt,
                response=response.content,
                model=response.model,
                tokens=response.tokens_used,
                latency_ms=response.latency_ms,
                context=context,
            )
            await collector.close()

    async_run(run())


@app.command()
def datalake(
    action: str = typer.Argument("stats", help="Action: stats, search, decisions, learnings"),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Search query"),
):
    """Explore connected datalakes."""
    from src.knowledge import DatalakeConnector

    connector = DatalakeConnector()

    async def run():
        if action == "stats":
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("Escaneando datalakes...", total=None)
                stats = await connector.get_stats()

            console.print(Panel.fit("[bold]Estadísticas de Datalakes[/bold]"))

            # Summary table
            table = Table(title="Resumen")
            table.add_column("Métrica", style="cyan")
            table.add_column("Valor", style="green")
            table.add_row("Total archivos", str(stats["total_files"]))
            table.add_row("Tamaño total", f"{stats['total_size_mb']:.2f} MB")
            console.print(table)

            # By datalake
            if stats["by_datalake"]:
                table = Table(title="Por Datalake")
                table.add_column("Datalake", style="cyan")
                table.add_column("Archivos", style="green")
                for name, count in sorted(stats["by_datalake"].items(), key=lambda x: -x[1]):
                    table.add_row(name, str(count))
                console.print(table)

            # By category
            if stats["by_category"]:
                table = Table(title="Por Categoría")
                table.add_column("Categoría", style="cyan")
                table.add_column("Archivos", style="green")
                for cat, count in sorted(stats["by_category"].items(), key=lambda x: -x[1]):
                    table.add_row(cat, str(count))
                console.print(table)

        elif action == "search" and query:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("Buscando...", total=None)
                results = await connector.search_files(query, limit=20)

            console.print(f"\n[bold]Resultados para '{query}':[/bold] {len(results)} encontrados\n")
            for f in results:
                console.print(f"[cyan]{f.datalake}[/cyan] / [dim]{f.relative_path}[/dim]")
                console.print(f"  [dim]Categoría: {f.category} | Tamaño: {f.size} bytes[/dim]")

        elif action == "decisions":
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("Cargando decisiones...", total=None)
                decisions = await connector.get_decisions()

            console.print(f"\n[bold]Decisiones técnicas:[/bold] {len(decisions)}\n")
            for d in decisions[:10]:
                console.print(f"[cyan]• {d.relative_path}[/cyan]")

        elif action == "learnings":
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("Cargando learnings...", total=None)
                learnings = await connector.get_learnings()

            console.print(f"\n[bold]Learnings:[/bold] {len(learnings)}\n")
            for l in learnings[:10]:
                console.print(f"[cyan]• {l.relative_path}[/cyan]")

        else:
            console.print("[yellow]Acciones disponibles: stats, search, decisions, learnings[/yellow]")

    async_run(run())


@app.command()
def flywheel(
    action: str = typer.Argument("stats", help="Action: stats, export, flush"),
):
    """Manage the data flywheel."""
    from src.flywheel import get_collector

    collector = get_collector()

    async def run():
        if action == "stats":
            stats = await collector.get_session_stats()
            console.print(Panel.fit("[bold]Flywheel Status[/bold]"))

            table = Table()
            table.add_column("Propiedad", style="cyan")
            table.add_column("Valor", style="green")
            table.add_row("Habilitado", "✓" if stats["enabled"] else "✗")
            table.add_row("Session ID", stats["session_id"][:8])
            table.add_row("En buffer", str(stats["buffered_records"]))
            console.print(table)

        elif action == "export":
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("Exportando training pairs...", total=None)
                output = await collector.export_training_pairs()

            console.print(f"[green]Exportado a:[/green] {output}")

        elif action == "flush":
            await collector.flush()
            console.print("[green]Buffer vaciado[/green]")

    async_run(run())


@app.command()
def rag(
    action: str = typer.Argument("stats", help="Action: index, search, stats"),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Search query"),
):
    """RAG system - index and search knowledge base."""
    from src.knowledge.rag import RAGEngine

    async def run():
        async with RAGEngine() as engine:
            if action == "index":
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True,
                ) as progress:
                    progress.add_task("Indexando datalake...", total=None)
                    stats = await engine.index_datalake()

                console.print(Panel.fit("[bold green]RAG Indexing Complete[/bold green]"))
                table = Table()
                table.add_column("Métrica", style="cyan")
                table.add_column("Valor", style="green")
                table.add_row("Archivos indexados", str(stats["files_indexed"]))
                table.add_row("Chunks creados", str(stats["chunks_created"]))
                table.add_row("Errores", str(stats["errors"]))
                console.print(table)

            elif action == "search" and query:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True,
                ) as progress:
                    progress.add_task("Buscando...", total=None)
                    results = await engine.retrieve(query, limit=5)

                console.print(f"\n[bold]Resultados para:[/bold] {query}\n")
                for i, r in enumerate(results, 1):
                    console.print(f"[cyan]{i}. [{r['category']}][/cyan] (score: {r['score']:.3f})")
                    console.print(f"   [dim]{r['text'][:200]}...[/dim]\n")

            elif action == "stats":
                stats = engine.get_stats()
                console.print(Panel.fit("[bold]RAG Stats[/bold]"))
                table = Table()
                table.add_column("Métrica", style="cyan")
                table.add_column("Valor", style="green")
                table.add_row("Documentos indexados", str(stats["total_documents"]))
                table.add_row("DB Path", str(stats.get("db_path", "N/A")))
                console.print(table)

            else:
                console.print("[yellow]Uso: rag index | rag search -q 'query' | rag stats[/yellow]")

    async_run(run())


@app.command()
def graph(
    action: str = typer.Argument("stats", help="Action: build, search, stats"),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Search query"),
    depth: int = typer.Option(2, "--depth", "-d", help="Traversal depth for search"),
    use_llm: bool = typer.Option(False, "--use-llm", help="Enable LLM extraction (requires Ollama)"),
    force: bool = typer.Option(False, "--force", help="Force rebuild from scratch"),
    include_transcripts: bool = typer.Option(
        False, "--include-transcripts",
        help="Include reasoning from Claude Code session transcripts",
    ),
):
    """Knowledge Graph - build, search, and inspect."""
    from src.knowledge.graph_engine import GraphEngine
    from src.knowledge.graph_schema import make_entity_id

    engine = GraphEngine()

    if action == "build":
        from src.knowledge.extraction.pipeline import ExtractionPipeline

        async def run_build():
            pipeline = ExtractionPipeline(
                engine=engine,
                use_llm=use_llm,
                include_transcripts=include_transcripts,
            )
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("Construyendo Knowledge Graph...", total=None)
                stats = await pipeline.build(force=force)

            console.print(Panel.fit("[bold green]Knowledge Graph Build Complete[/bold green]"))
            table = Table()
            table.add_column("Metrica", style="cyan")
            table.add_column("Valor", style="green")
            table.add_row("Archivos procesados", str(stats["files_processed"]))
            table.add_row("Pairs procesados", str(stats["pairs_processed"]))
            table.add_row("Triples extraidos", str(stats["triples_extracted"]))
            table.add_row("Errores", str(stats["errors"]))
            if stats.get("transcript_triples_extracted", 0) > 0:
                table.add_row(
                    "Triples de transcripts",
                    str(stats["transcript_triples_extracted"]),
                )
            if stats.get("inferred_triples", 0) > 0:
                table.add_row(
                    "Triples inferidos",
                    str(stats["inferred_triples"]),
                )
            console.print(table)

            graph_stats = engine.get_stats()
            console.print(f"\n[dim]Entidades: {graph_stats['entity_count']} | "
                          f"Relaciones: {graph_stats['edge_count']}[/dim]")

        async_run(run_build())

    elif action == "search" and query:
        engine.load()
        results = engine.search_entities(query, limit=10)

        if not results:
            console.print(f"[yellow]No se encontraron entidades para '{query}'[/yellow]")
            return

        console.print(f"\n[bold]Entidades para '{query}':[/bold]\n")
        for entity in results:
            console.print(
                f"[cyan]{entity.name}[/cyan] "
                f"[dim]({entity.entity_type.value})[/dim] "
                f"mentions={entity.mention_count}"
            )

            if entity.aliases:
                console.print(f"  [dim]aliases: {', '.join(entity.aliases)}[/dim]")

            # Show neighbors
            neighbors = engine.get_neighbors(entity.id, depth=depth, min_weight=0.3)
            if neighbors:
                neighbor_strs = []
                for n, score in neighbors[:5]:
                    rel_info = ""
                    rels = engine.get_relations(entity.id, direction="out")
                    for r in rels:
                        if r.target_id == n.id:
                            rel_info = f" [{r.relation_type.value}]"
                            break
                    neighbor_strs.append(f"{n.name}{rel_info}")
                console.print(f"  [green]-> {', '.join(neighbor_strs)}[/green]")
            console.print()

    elif action == "stats":
        if engine.load():
            stats = engine.get_stats()
            console.print(Panel.fit("[bold]Knowledge Graph Stats[/bold]"))

            table = Table()
            table.add_column("Metrica", style="cyan")
            table.add_column("Valor", style="green")
            table.add_row("Entidades", str(stats["entity_count"]))
            table.add_row("Relaciones", str(stats["edge_count"]))
            table.add_row("Componentes conexos", str(stats["connected_components"]))
            table.add_row("Graph path", str(stats["graph_path"]))
            console.print(table)

            if stats["entity_types"]:
                table = Table(title="Por tipo de entidad")
                table.add_column("Tipo", style="cyan")
                table.add_column("Count", style="green")
                for etype, count in sorted(stats["entity_types"].items(), key=lambda x: -x[1]):
                    table.add_row(etype, str(count))
                console.print(table)

            if stats["relation_types"]:
                table = Table(title="Por tipo de relacion")
                table.add_column("Tipo", style="cyan")
                table.add_column("Count", style="green")
                for rtype, count in sorted(stats["relation_types"].items(), key=lambda x: -x[1]):
                    table.add_row(rtype, str(count))
                console.print(table)
        else:
            console.print("[yellow]No hay Knowledge Graph construido.[/yellow]")
            console.print("[dim]Ejecuta: fabrik graph build[/dim]")

    elif action == "complete":
        if not engine.load():
            console.print("[yellow]No hay Knowledge Graph construido.[/yellow]")
            console.print("[dim]Ejecuta: fabrik graph build[/dim]")
            return

        stats = engine.complete()
        engine.save()

        console.print(Panel.fit("[bold green]Graph Completion Done[/bold green]"))
        table = Table()
        table.add_column("Metrica", style="cyan")
        table.add_column("Valor", style="green")
        table.add_row("Total inferidos", str(stats["inferred_count"]))
        table.add_row("DEPENDS_ON inferidos", str(stats["depends_on_inferred"]))
        table.add_row("PART_OF inferidos", str(stats["part_of_inferred"]))
        console.print(table)

    else:
        console.print("[yellow]Uso: graph build | graph search -q 'query' | graph stats | graph complete[/yellow]")


@app.command()
def learn(
    action: str = typer.Argument("process", help="Action: process, stats, reset"),
):
    """Learn from Claude Code sessions - extract training data."""
    from src.flywheel.session_observer import process_all_sessions, get_stats, PROCESSED_MARKER

    if action == "process":
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Procesando sesiones de Claude Code...", total=None)
            stats = process_all_sessions()

        console.print(Panel.fit("[bold green]Session Observer[/bold green]"))

        table = Table()
        table.add_column("Métrica", style="cyan")
        table.add_column("Valor", style="green")
        table.add_row("Sesiones procesadas", str(stats["sessions_processed"]))
        table.add_row("Training pairs extraídos", str(stats["pairs_extracted"]))
        console.print(table)

        if stats.get("by_category"):
            table = Table(title="Por categoría")
            table.add_column("Categoría", style="cyan")
            table.add_column("Count", style="green")
            for cat, count in sorted(stats["by_category"].items(), key=lambda x: -x[1]):
                table.add_row(cat, str(count))
            console.print(table)

        if stats.get("output_file"):
            console.print(f"\n[dim]Output: {stats['output_file']}[/dim]")

    elif action == "stats":
        stats = get_stats()
        console.print(Panel.fit("[bold]Learning Stats[/bold]"))

        table = Table()
        table.add_column("Métrica", style="cyan")
        table.add_column("Valor", style="green")
        table.add_row("Sesiones procesadas", str(stats["sessions_processed"]))
        table.add_row("Total training pairs", str(stats["total_training_pairs"]))
        console.print(table)

    elif action == "reset":
        if PROCESSED_MARKER.exists():
            PROCESSED_MARKER.unlink()
            console.print("[green]Reset completo. Todas las sesiones serán reprocesadas.[/green]")
        else:
            console.print("[yellow]Nada que resetear.[/yellow]")


@app.command()
def finetune(
    epochs: int = typer.Option(3, "--epochs", "-e", help="Training epochs"),
    batch_size: int = typer.Option(2, "--batch-size", "-b", help="Batch size"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", "-n", help="Limit training samples"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show stats without training"),
):
    """Fine-tune Qwen with your training data."""
    import subprocess

    if dry_run:
        # Just show training data stats
        from src.flywheel.session_observer import get_stats
        stats = get_stats()

        console.print(Panel.fit("[bold]Fine-tuning Data[/bold]"))
        table = Table()
        table.add_column("Métrica", style="cyan")
        table.add_column("Valor", style="green")
        table.add_row("Training pairs disponibles", str(stats["total_training_pairs"]))
        table.add_row("Épocas configuradas", str(epochs))
        table.add_row("Batch size", str(batch_size))
        console.print(table)

        console.print("\n[dim]Para entrenar ejecuta sin --dry-run[/dim]")
        return

    console.print(Panel.fit("[bold yellow]Starting Fine-tuning[/bold yellow]"))
    console.print("[dim]Esto puede tomar varias horas...[/dim]\n")

    # Run finetune script
    cmd = [
        "python", "scripts/finetune.py",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
    ]
    if max_samples:
        cmd.extend(["--max-samples", str(max_samples)])

    subprocess.run(cmd)


@app.command()
def models():
    """List available Ollama models."""
    from src.core import LLMClient

    async def run():
        async with LLMClient() as client:
            if not await client.health_check():
                console.print("[red]Ollama no disponible[/red]")
                return

            models = await client.list_models()

            table = Table(title="Modelos disponibles")
            table.add_column("Modelo", style="cyan")
            for m in models:
                table.add_row(m)
            console.print(table)

    async_run(run())


@app.command()
def serve(
    host: Optional[str] = typer.Option(None, "--host", "-h", help="Bind host"),
    port: Optional[int] = typer.Option(None, "--port", "-p", help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes"),
):
    """Start the Fabrik-Codek API server."""
    import uvicorn

    from src.config import settings

    bind_host = host or settings.api_host
    bind_port = port or settings.api_port

    console.print(
        Panel.fit(
            f"[bold blue]Fabrik-Codek API[/bold blue]\n"
            f"http://{bind_host}:{bind_port}",
            subtitle="Ctrl+C to stop",
        )
    )

    uvicorn.run(
        "src.interfaces.api:app",
        host=bind_host,
        port=bind_port,
        reload=reload,
    )


@app.command()
def status():
    """Check system status."""
    from src.config import settings
    from src.core import LLMClient

    console.print(Panel.fit("[bold blue]Fabrik-Codek Status[/bold blue]"))

    table = Table()
    table.add_column("Componente", style="cyan")
    table.add_column("Estado", style="green")
    table.add_column("Detalle", style="dim")

    # Settings
    table.add_row("Config", "✓", f"Model: {settings.default_model}")

    # Datalake
    datalake_exists = settings.datalake_path.exists()
    table.add_row(
        "Datalake",
        "✓" if datalake_exists else "✗",
        str(settings.datalake_path)[:50],
    )

    # Flywheel
    table.add_row(
        "Flywheel",
        "✓" if settings.flywheel_enabled else "✗",
        f"Batch: {settings.flywheel_batch_size}",
    )

    # Knowledge Graph
    from src.knowledge.graph_engine import GraphEngine
    graph_engine = GraphEngine()
    if graph_engine.load():
        gstats = graph_engine.get_stats()
        table.add_row(
            "Knowledge Graph",
            "✓",
            f"{gstats['entity_count']} entities, {gstats['edge_count']} edges",
        )
    else:
        table.add_row("Knowledge Graph", "✗", "No construido (fabrik graph build)")

    console.print(table)

    # Ollama check
    async def check_ollama():
        try:
            async with LLMClient() as client:
                ok = await client.health_check()
                return ok
        except Exception:
            return False

    ollama_ok = async_run(check_ollama())
    if ollama_ok:
        console.print("\n[green]✓ Ollama conectado[/green]")
    else:
        console.print("\n[red]✗ Ollama no disponible[/red]")
        console.print("[dim]  Ejecuta: ollama serve[/dim]")


@app.command()
def init(
    skip_models: bool = typer.Option(False, "--skip-models", help="Skip downloading Ollama models"),
):
    """Initialize Fabrik-Codek: check dependencies, create config, download models."""
    import shutil
    import subprocess
    import sys
    import urllib.request

    console.print(Panel.fit(
        "[bold blue]Fabrik-Codek Setup[/bold blue] - Claude's Little Brother",
        subtitle="Initializing...",
    ))
    console.print()

    errors = []

    # 1. Check Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 11):
        console.print(f"[green]✓[/green] Python {py_version}")
    else:
        console.print(f"[red]✗[/red] Python {py_version} (3.11+ required)")
        errors.append("Python 3.11+ is required")

    # 2. Check Ollama
    ollama_installed = shutil.which("ollama") is not None
    ollama_running = False

    if ollama_installed:
        console.print("[green]✓[/green] Ollama installed")
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                ollama_running = resp.status == 200
        except Exception:
            ollama_running = False

        if ollama_running:
            console.print("[green]✓[/green] Ollama server running")
        else:
            console.print("[yellow]![/yellow] Ollama not running. Start with: [bold]ollama serve[/bold]")
    else:
        console.print("[yellow]![/yellow] Ollama not installed")
        console.print("  Install: [bold]curl -fsSL https://ollama.com/install.sh | sh[/bold]")

    # 3. Create .env if not exists
    console.print()
    from src.config.settings import Settings

    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / ".env"
    env_example = project_root / ".env.example"

    if env_file.exists():
        console.print("[green]✓[/green] .env file exists")
    elif env_example.exists():
        shutil.copy2(env_example, env_file)
        console.print("[green]✓[/green] .env created from .env.example")
    else:
        env_file.write_text(
            "# Fabrik-Codek Configuration\n"
            "FABRIK_OLLAMA_HOST=http://localhost:11434\n"
            "FABRIK_DEFAULT_MODEL=qwen2.5-coder:7b\n"
            "FABRIK_FLYWHEEL_ENABLED=true\n"
            "FABRIK_LOG_LEVEL=INFO\n"
        )
        console.print("[green]✓[/green] .env file created")

    # 4. Create data directories
    data_dir = project_root / "data"
    dirs_created = 0
    for subdir in ["raw", "processed", "embeddings", "raw/interactions", "raw/training_pairs"]:
        d = data_dir / subdir
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            dirs_created += 1

    if dirs_created > 0:
        console.print(f"[green]✓[/green] Data directories created ({dirs_created} new)")
    else:
        console.print("[green]✓[/green] Data directories exist")

    # 5. Download models if Ollama is running
    if ollama_running and not skip_models:
        console.print()
        models_to_check = ["qwen2.5-coder:7b", "nomic-embed-text"]

        for model_name in models_to_check:
            # Check if model already exists
            try:
                result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True, text=True, timeout=10,
                )
                if model_name in result.stdout:
                    console.print(f"[green]✓[/green] {model_name} available")
                    continue
            except Exception:
                pass

            console.print(f"[yellow]![/yellow] Downloading {model_name}...")
            try:
                subprocess.run(
                    ["ollama", "pull", model_name],
                    timeout=600,
                    check=True,
                )
                console.print(f"[green]✓[/green] {model_name} downloaded")
            except subprocess.TimeoutExpired:
                console.print(f"[red]✗[/red] {model_name} download timed out")
            except subprocess.CalledProcessError:
                console.print(f"[red]✗[/red] Failed to download {model_name}")
    elif not ollama_running and not skip_models:
        console.print()
        console.print("[yellow]![/yellow] Skipping model download (Ollama not running)")
        console.print("  After starting Ollama, run:")
        console.print("    [bold]ollama pull qwen2.5-coder:7b[/bold]")
        console.print("    [bold]ollama pull nomic-embed-text[/bold]")

    # 6. Summary
    console.print()
    if errors:
        console.print(Panel.fit(
            "[bold red]Setup incomplete[/bold red]\n" +
            "\n".join(f"  • {e}" for e in errors),
        ))
        raise typer.Exit(code=1)

    console.print(Panel.fit(
        "[bold green]Setup Complete![/bold green]\n\n"
        "  [bold]fabrik status[/bold]     Check system health\n"
        "  [bold]fabrik chat[/bold]       Start interactive chat\n"
        "  [bold]fabrik ask \"...\"[/bold]  Ask a single question\n"
        "  [bold]fabrik mcp[/bold]        Start as MCP server",
        subtitle="Ready to go",
    ))


@app.command()
def mcp(
    transport: str = typer.Option("stdio", "--transport", "-t", help="Transport: stdio or sse"),
    port: Optional[int] = typer.Option(None, "--port", "-p", help="Port for SSE transport"),
):
    """Start Fabrik-Codek as an MCP server for agent integration."""
    from src.interfaces.mcp_server import mcp as mcp_server

    if transport == "sse":
        from src.config import settings
        bind_port = port or settings.mcp_port
        console.print(
            Panel.fit(
                f"[bold blue]Fabrik-Codek MCP Server[/bold blue] (SSE)\n"
                f"http://127.0.0.1:{bind_port}/sse",
                subtitle="Ctrl+C to stop",
            )
        )
        mcp_server.run(transport="sse", port=bind_port)
    else:
        mcp_server.run(transport="stdio")


@app.callback()
def main():
    """Fabrik-Codek: Claude Code's little brother 🤖"""
    pass


if __name__ == "__main__":
    app()
