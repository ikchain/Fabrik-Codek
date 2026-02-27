"""Fabrik-Codek CLI - Your local AI dev assistant."""

import asyncio
from pathlib import Path
from uuid import uuid4

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

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
    model: str | None = typer.Option(None, "--model", "-m", help="Model to use"),
    system: str | None = typer.Option(None, "--system", "-s", help="System prompt"),
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

            # Route via TaskRouter for adaptive system prompt
            from src.core.competence_model import get_active_competence_map
            from src.core.personal_profile import get_active_profile
            from src.core.task_router import TaskRouter, load_learned_classifier

            active_profile = get_active_profile()
            competence_map = get_active_competence_map()
            from src.core.strategy_optimizer import MABStrategyOptimizer

            mab = MABStrategyOptimizer(settings.data_dir)
            learned = load_learned_classifier(settings)
            router = TaskRouter(
                competence_map, active_profile, settings, mab=mab, learned_classifier=learned
            )

            from src.flywheel.collector import bridge_outcome_to_feedback
            from src.flywheel.outcome_tracker import OutcomeTracker

            tracker = OutcomeTracker(settings.datalake_path, str(uuid4()))

            initial_decision = await router.route("general conversation")
            if not system:
                messages.insert(0, {"role": "system", "content": initial_decision.system_prompt})

            last_record_id = None

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

                # Bridge: outcome -> feedback on the PREVIOUS record
                if prev_outcome is not None and last_record_id is not None:
                    try:
                        await bridge_outcome_to_feedback(
                            collector,
                            prev_outcome,
                            last_record_id,
                        )
                    except Exception:
                        pass  # Non-critical: never crash chat for feedback

                # Update MAB with outcome feedback (FC-42)
                if prev_outcome is not None and prev_outcome.outcome != "neutral":
                    arm_id = prev_outcome.strategy.get("arm_id")
                    if arm_id:
                        mab.update(
                            task_type=prev_outcome.task_type,
                            topic=prev_outcome.topic,
                            arm_id=arm_id,
                            reward=1.0 if prev_outcome.outcome == "accepted" else 0.0,
                        )

                last_record_id = record.id

        mab.save_state()
        tracker.close_session()
        stats = tracker.get_session_stats()
        if stats["total"] > 0:
            console.print(
                f"[dim]Outcomes: {stats['total']} tracked "
                f"({stats['accepted']} accepted, "
                f"{stats['rejected']} rejected)[/dim]"
            )

        await collector.close()

    async_run(run_chat())
    console.print("\n[dim]¡Hasta luego! 👋[/dim]")


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="Question or task"),
    model: str | None = typer.Option(None, "--model", "-m"),
    context_file: Path | None = typer.Option(None, "--context", "-c", help="File for context"),
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

        # Route via TaskRouter for adaptive prompt/model/strategy
        from src.config import settings
        from src.core.competence_model import get_active_competence_map
        from src.core.personal_profile import get_active_profile
        from src.core.task_router import TaskRouter, load_learned_classifier

        active_profile = get_active_profile()
        competence_map = get_active_competence_map()
        from src.core.strategy_optimizer import MABStrategyOptimizer

        mab = MABStrategyOptimizer(settings.data_dir)
        learned = load_learned_classifier(settings)
        router = TaskRouter(
            competence_map, active_profile, settings, mab=mab, learned_classifier=learned
        )
        decision = await router.route(prompt)

        console.print(
            f"[dim]Router: {decision.task_type} "
            f"({decision.classification_method}) "
            f"| topic={decision.topic or '—'} "
            f"| competence={decision.competence_level} "
            f"| model={decision.model}[/dim]"
        )

        # Inject hybrid RAG context (vector + graph)
        if use_graph:
            from src.knowledge.hybrid_rag import HybridRAGEngine

            async with HybridRAGEngine() as hybrid:
                results = await hybrid.retrieve(
                    prompt,
                    limit=5,
                    graph_depth=decision.strategy.graph_depth,
                )
                if results:
                    final_prompt = await hybrid.query_with_context(
                        prompt,
                        limit=5,
                        graph_depth=decision.strategy.graph_depth,
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
                    rag_context = "\n---\n".join(
                        [f"[{r['category']}] {r['text'][:500]}" for r in rag_results]
                    )
                    final_prompt = f"""Contexto de tu base de conocimiento:
{rag_context}

---
Pregunta: {prompt}

Responde usando el contexto cuando sea relevante."""
                    context = rag_context
                    console.print(
                        f"[dim]RAG: {len(rag_results)} documentos relevantes encontrados[/dim]\n"
                    )

        async with LLMClient(model=model) as client:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("Procesando...", total=None)
                response = await client.generate(
                    final_prompt,
                    system=decision.system_prompt,
                    model=decision.model,
                )

            console.print(Markdown(response.content))
            console.print(
                f"\n[dim]({response.tokens_used} tokens, {response.latency_ms:.0f}ms)[/dim]"
            )

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

            # Track outcome
            from src.flywheel.outcome_tracker import OutcomeTracker

            ot = OutcomeTracker(settings.datalake_path, str(uuid4()))
            ot.record_turn(
                query=prompt,
                response=response.content,
                decision=decision,
                latency_ms=response.latency_ms,
            )
            ot.close_session()
            mab.save_state()

    async_run(run())


@app.command()
def datalake(
    action: str = typer.Argument("stats", help="Action: stats, search, decisions, learnings"),
    query: str | None = typer.Option(None, "--query", "-q", help="Search query"),
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
            for learning in learnings[:10]:
                console.print(f"[cyan]• {learning.relative_path}[/cyan]")

        else:
            console.print(
                "[yellow]Acciones disponibles: stats, search, decisions, learnings[/yellow]"
            )

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
    query: str | None = typer.Option(None, "--query", "-q", help="Search query"),
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
    action: str = typer.Argument(
        "stats", help="Action: build, search, stats, complete, prune, decay, aliases, drift"
    ),
    query: str | None = typer.Option(None, "--query", "-q", help="Search query"),
    depth: int = typer.Option(2, "--depth", "-d", help="Traversal depth for search"),
    use_llm: bool = typer.Option(
        False, "--use-llm", help="Enable LLM extraction (requires Ollama)"
    ),
    force: bool = typer.Option(False, "--force", help="Force rebuild from scratch"),
    include_transcripts: bool = typer.Option(
        False,
        "--include-transcripts",
        help="Include reasoning from Claude Code session transcripts",
    ),
    min_mentions: int = typer.Option(
        1, "--min-mentions", help="Min mention_count to keep isolated entities"
    ),
    min_weight: float = typer.Option(0.3, "--min-weight", help="Min edge weight to keep"),
    keep_inferred: bool = typer.Option(
        False, "--keep-inferred", help="Preserve inferred edges during prune"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without modifying"),
    half_life: float = typer.Option(90.0, "--half-life", help="Decay half-life in days"),
    threshold: float = typer.Option(
        0.85, "--threshold", help="Similarity threshold for alias detection"
    ),
    apply: bool = typer.Option(False, "--apply", help="Apply alias merges (default: dry-run)"),
    deduplicate: bool = typer.Option(
        False, "--deduplicate", help="Run alias deduplication after build"
    ),
):
    """Knowledge Graph - build, search, inspect, and deduplicate."""
    from src.knowledge.graph_engine import GraphEngine

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
                stats = await pipeline.build(
                    force=force,
                    deduplicate=deduplicate,
                )

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
            if stats.get("aliases_candidates", 0) > 0:
                table.add_row(
                    "Alias candidatos",
                    str(stats["aliases_candidates"]),
                )
                table.add_row(
                    "Aliases merged",
                    str(stats.get("aliases_merged", 0)),
                )
            console.print(table)

            graph_stats = engine.get_stats()
            console.print(
                f"\n[dim]Entidades: {graph_stats['entity_count']} | "
                f"Relaciones: {graph_stats['edge_count']}[/dim]"
            )

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

    elif action == "prune":
        if not engine.load():
            console.print("[yellow]No hay Knowledge Graph construido.[/yellow]")
            console.print("[dim]Ejecuta: fabrik graph build[/dim]")
            return

        before_stats = engine.get_stats()
        result = engine.prune(
            min_mention_count=min_mentions,
            min_edge_weight=min_weight,
            keep_inferred=keep_inferred,
            dry_run=dry_run,
        )

        if not dry_run:
            engine.save()

        title = (
            "[bold yellow]Graph Prune Preview (dry-run)[/bold yellow]"
            if dry_run
            else "[bold green]Graph Prune Complete[/bold green]"
        )
        console.print(Panel.fit(title))

        table = Table()
        table.add_column("Metrica", style="cyan")
        table.add_column("Valor", style="green")
        table.add_row("Entidades antes", str(before_stats["entity_count"]))
        table.add_row("Edges antes", str(before_stats["edge_count"]))
        table.add_row("Edges eliminados", str(result["edges_removed"]))
        table.add_row("Entidades eliminadas", str(result["entities_removed"]))
        console.print(table)

        if dry_run and result["removed_entities"]:
            console.print("\n[bold]Entidades a eliminar:[/bold]")
            for ent in result["removed_entities"][:20]:
                console.print(f"  [dim]{ent['name']}[/dim] ({ent['type']})")
            if len(result["removed_entities"]) > 20:
                console.print(f"  [dim]... y {len(result['removed_entities']) - 20} mas[/dim]")

    elif action == "decay":
        if not engine.load():
            console.print("[yellow]No hay Knowledge Graph construido.[/yellow]")
            console.print("[dim]Ejecuta: fabrik graph build[/dim]")
            return

        before_stats = engine.get_stats()
        result = engine.apply_decay(half_life_days=half_life, dry_run=dry_run)

        if not dry_run:
            engine.save()

        title = (
            "[bold yellow]Graph Decay Preview (dry-run)[/bold yellow]"
            if dry_run
            else "[bold green]Graph Decay Applied[/bold green]"
        )
        console.print(Panel.fit(title))

        table = Table()
        table.add_column("Metrica", style="cyan")
        table.add_column("Valor", style="green")
        table.add_row("Edges totales", str(before_stats["edge_count"]))
        table.add_row("Edges decayed", str(result["edges_decayed"]))
        table.add_row("Edges skipped (legacy)", str(result["edges_skipped"]))
        table.add_row("Min weight after", f"{result['min_weight_after']:.6f}")
        table.add_row("Max weight after", f"{result['max_weight_after']:.6f}")
        table.add_row("Half-life", f"{half_life} days")
        console.print(table)

    elif action == "aliases":
        if not engine.load():
            console.print("[yellow]No hay Knowledge Graph construido.[/yellow]")
            console.print("[dim]Ejecuta: fabrik graph build[/dim]")
            return

        import httpx

        from src.config import settings as fabrik_settings

        async def run_aliases():
            entity_names = {eid: entity.name for eid, entity in engine._entities.items()}

            if not entity_names:
                console.print("[yellow]El grafo no tiene entidades.[/yellow]")
                return

            # Compute embeddings via Ollama
            embeddings: dict[str, list[float]] = {}
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task(
                    f"Generando embeddings para {len(entity_names)} entidades...",
                    total=None,
                )
                async with httpx.AsyncClient(timeout=60.0) as client:
                    for eid, name in entity_names.items():
                        try:
                            resp = await client.post(
                                f"{fabrik_settings.ollama_host}/api/embeddings",
                                json={
                                    "model": fabrik_settings.embedding_model,
                                    "prompt": name,
                                },
                            )
                            resp.raise_for_status()
                            embeddings[eid] = resp.json()["embedding"]
                        except (httpx.HTTPError, KeyError, OSError):
                            pass  # Skip entities that fail to embed

            if not embeddings:
                console.print(
                    "[red]No se pudieron generar embeddings. Verifica que Ollama esta corriendo.[/red]"
                )
                return

            is_dry_run = not apply
            result = engine.deduplicate_aliases(
                embeddings,
                threshold=threshold,
                dry_run=is_dry_run,
            )

            if not is_dry_run:
                engine.save()

            title = (
                "[bold yellow]Alias Detection Preview (dry-run)[/bold yellow]"
                if is_dry_run
                else "[bold green]Alias Deduplication Applied[/bold green]"
            )
            console.print(Panel.fit(title))

            # Show pairs table
            if result["pairs"]:
                pairs_table = Table(title="Alias Pairs Detected")
                pairs_table.add_column("Canonical", style="cyan")
                pairs_table.add_column("Alias", style="yellow")
                pairs_table.add_column("Similarity", style="green")
                for canonical, alias_name, sim in result["pairs"]:
                    pairs_table.add_row(canonical, alias_name, f"{sim:.4f}")
                console.print(pairs_table)

            # Summary stats
            summary = Table()
            summary.add_column("Metrica", style="cyan")
            summary.add_column("Valor", style="green")
            summary.add_row("Entidades analizadas", str(len(embeddings)))
            summary.add_row("Candidatos detectados", str(result["candidates"]))
            summary.add_row("Merges aplicados", str(result["merged"]))
            summary.add_row("Threshold", f"{threshold:.2f}")
            console.print(summary)

        async_run(run_aliases())

    elif action == "drift":
        if not engine.load():
            console.print("[yellow]No hay Knowledge Graph construido.[/yellow]")
            console.print("[dim]Ejecuta: fabrik graph build[/dim]")
            return

        if dry_run:
            # Live detection: compare current state vs stored snapshots
            events = engine.detect_drift(threshold=threshold)
            if events:
                console.print(
                    Panel.fit(
                        f"[bold yellow]Drift Detection Preview: {len(events)} entities[/bold yellow]"
                    )
                )
                drift_table = Table()
                drift_table.add_column("Entity", style="cyan")
                drift_table.add_column("Type", style="dim")
                drift_table.add_column("Jaccard", style="yellow")
                drift_table.add_column("Added", style="green")
                drift_table.add_column("Removed", style="red")
                for ev in events[:20]:
                    drift_table.add_row(
                        ev.entity_name,
                        ev.entity_type,
                        f"{ev.jaccard_similarity:.2f}",
                        ", ".join(ev.added[:3]) or "-",
                        ", ".join(ev.removed[:3]) or "-",
                    )
                console.print(drift_table)
            else:
                console.print("[green]No semantic drift detected.[/green]")
        else:
            # Show drift log history
            entries = engine.load_drift_log(entity_name=query)
            if entries:
                console.print(Panel.fit(f"[bold]Drift Log: {len(entries)} events[/bold]"))
                drift_table = Table()
                drift_table.add_column("Timestamp", style="dim")
                drift_table.add_column("Entity", style="cyan")
                drift_table.add_column("Jaccard", style="yellow")
                drift_table.add_column("Added", style="green")
                drift_table.add_column("Removed", style="red")
                for entry in entries[-20:]:
                    drift_table.add_row(
                        entry.get("timestamp", "")[:19],
                        entry.get("entity_name", ""),
                        f"{entry.get('jaccard_similarity', 0):.2f}",
                        ", ".join(entry.get("added", [])[:3]) or "-",
                        ", ".join(entry.get("removed", [])[:3]) or "-",
                    )
                console.print(drift_table)
            else:
                console.print("[dim]No drift events in log.[/dim]")

    else:
        console.print(
            "[yellow]Uso: graph build | search | stats | complete | prune | decay | aliases | drift[/yellow]"
        )


@app.command()
def learn(
    action: str = typer.Argument("process", help="Action: process, stats, reset, watch"),
    interval: int = typer.Option(60, help="Seconds between watch cycles"),
):
    """Learn from Claude Code sessions - extract training data."""
    from src.flywheel.session_observer import PROCESSED_MARKER, get_stats, process_all_sessions

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

    elif action == "watch":
        import signal

        console.print(f"[bold]Watching sessions every {interval}s... (Ctrl+C to stop)[/bold]")

        async def run_watch():
            stop = asyncio.Event()

            def handle_signal():
                stop.set()

            loop = asyncio.get_event_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, handle_signal)

            while not stop.is_set():
                try:
                    stats = process_all_sessions(min_quality=0.4)
                    if stats["sessions_processed"] > 0:
                        console.print(
                            f"[green]Processed {stats['sessions_processed']} sessions, "
                            f"{stats['pairs_extracted']} pairs extracted[/green]"
                        )
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")

                try:
                    await asyncio.wait_for(stop.wait(), timeout=interval)
                except TimeoutError:
                    pass  # Normal: timeout expired, continue polling

            console.print("[yellow]Watch mode stopped.[/yellow]")

        asyncio.run(run_watch())


@app.command()
def finetune(
    epochs: int = typer.Option(3, "--epochs", "-e", help="Training epochs"),
    batch_size: int = typer.Option(2, "--batch-size", "-b", help="Batch size"),
    max_samples: int | None = typer.Option(
        None, "--max-samples", "-n", help="Limit training samples"
    ),
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
        "python",
        "scripts/finetune.py",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
    ]
    if max_samples:
        cmd.extend(["--max-samples", str(max_samples)])

    project_root = str(Path(__file__).parent.parent.parent)
    subprocess.run(cmd, cwd=project_root)


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
    host: str | None = typer.Option(None, "--host", "-h", help="Bind host"),
    port: int | None = typer.Option(None, "--port", "-p", help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes"),
):
    """Start the Fabrik-Codek API server."""
    import uvicorn

    from src.config import settings

    bind_host = host or settings.api_host
    bind_port = port or settings.api_port

    console.print(
        Panel.fit(
            f"[bold blue]Fabrik-Codek API[/bold blue]\n" f"http://{bind_host}:{bind_port}",
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

    console.print(
        Panel.fit(
            "[bold blue]Fabrik-Codek Setup[/bold blue] - Claude's Little Brother",
            subtitle="Initializing...",
        )
    )
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
            console.print(
                "[yellow]![/yellow] Ollama not running. Start with: [bold]ollama serve[/bold]"
            )
    else:
        console.print("[yellow]![/yellow] Ollama not installed")
        console.print("  Install: [bold]curl -fsSL https://ollama.com/install.sh | sh[/bold]")

    # 3. Create .env if not exists
    console.print()

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
                    capture_output=True,
                    text=True,
                    timeout=10,
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
        console.print(
            Panel.fit(
                "[bold red]Setup incomplete[/bold red]\n" + "\n".join(f"  • {e}" for e in errors),
            )
        )
        raise typer.Exit(code=1)

    console.print(
        Panel.fit(
            "[bold green]Setup Complete![/bold green]\n\n"
            "  [bold]fabrik status[/bold]     Check system health\n"
            "  [bold]fabrik chat[/bold]       Start interactive chat\n"
            '  [bold]fabrik ask "..."[/bold]  Ask a single question\n'
            "  [bold]fabrik mcp[/bold]        Start as MCP server",
            subtitle="Ready to go",
        )
    )


@app.command()
def mcp(
    transport: str = typer.Option("stdio", "--transport", "-t", help="Transport: stdio or sse"),
    port: int | None = typer.Option(None, "--port", "-p", help="Port for SSE transport"),
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


@app.command()
def fulltext(
    action: str = typer.Argument("status", help="Action: status, index, search"),
    query: str | None = typer.Option(None, "--query", "-q", help="Search query"),
    limit: int = typer.Option(5, "--limit", "-n", help="Max results"),
):
    """Full-text search via Meilisearch - status, index datalake, or search."""
    from src.knowledge.fulltext_engine import FullTextEngine

    if action == "status":

        async def _check():
            async with FullTextEngine() as ft:
                healthy = await ft.health_check()
                if not healthy:
                    console.print("[yellow]Meilisearch:[/yellow] unavailable")
                    console.print(f"  URL: {ft._url}")
                    console.print("  Run: meilisearch --master-key=fabrik-dev-key")
                    return
                stats = await ft.get_stats()
                console.print("[green]Meilisearch:[/green] connected")
                console.print(f"  Documents: {stats['document_count']}")
                console.print(f"  Indexing:  {stats['is_indexing']}")

        async_run(_check())

    elif action == "index":
        import json as json_mod

        from src.config import settings

        async def _index():
            async with FullTextEngine() as ft:
                if not await ft.health_check():
                    console.print("[red]Error:[/red] Meilisearch not available")
                    raise typer.Exit(1)

                await ft.ensure_index()
                console.print("[blue]Indexing datalake into Meilisearch...[/blue]")

                tp_dir = settings.datalake_path / "02-processed" / "training-pairs"
                total = 0
                if tp_dir.exists():
                    files = sorted(tp_dir.glob("*.jsonl"))
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console,
                        transient=True,
                    ) as progress:
                        task = progress.add_task("Indexing...", total=len(files))
                        for f in files:
                            docs = []
                            for line in f.read_text().splitlines():
                                if not line.strip():
                                    continue
                                try:
                                    record = json_mod.loads(line)
                                    text = record.get("output", record.get("text", ""))
                                    instruction = record.get("instruction", "")
                                    if text and len(text) >= 50:
                                        docs.append(
                                            {
                                                "id": FullTextEngine.make_doc_id(text, f.name),
                                                "text": (
                                                    f"{instruction}\n{text}"
                                                    if instruction
                                                    else text
                                                ),
                                                "source": f.name,
                                                "category": "training",
                                                "project": record.get("project", ""),
                                            }
                                        )
                                except json_mod.JSONDecodeError:
                                    continue
                            if docs:
                                count = await ft.index_documents(docs)
                                total += count
                            progress.advance(task)

                console.print(f"[green]Indexed {total} documents[/green]")

        async_run(_index())

    elif action == "search":
        if not query:
            console.print("[red]Error:[/red] --query/-q is required for search")
            raise typer.Exit(1)

        async def _search():
            async with FullTextEngine() as ft:
                if not await ft.health_check():
                    console.print("[red]Error:[/red] Meilisearch not available")
                    raise typer.Exit(1)

                results = await ft.search(query, limit=limit)
                if not results:
                    console.print("[yellow]No results[/yellow]")
                    return

                for i, r in enumerate(results, 1):
                    console.print(f"\n[bold]{i}.[/bold] [{r['category']}] {r['source']}")
                    console.print(f"   {r['text'][:200]}")

        async_run(_search())

    else:
        console.print(f"[red]Unknown action:[/red] {action}")
        console.print("Available: status, index, search")
        raise typer.Exit(1)


@app.command()
def profile(
    action: str = typer.Argument("show", help="Action: show, build, build-incremental, drift"),
):
    """Manage your personal profile — learned from the datalake."""
    from src.config import settings
    from src.core.personal_profile import ProfileBuilder, load_profile

    profile_path = settings.data_dir / "profile" / "personal_profile.json"

    if action == "build":
        console.print(Panel.fit("[bold]Building Personal Profile...[/bold]"))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Analyzing datalake...", total=None)
            builder = ProfileBuilder(datalake_path=settings.datalake_path)
            result = builder.build(output_path=profile_path)

        table = Table(title="Profile Built")
        table.add_column("Field", style="bold cyan")
        table.add_column("Value", style="green")
        table.add_row("Domain", f"{result.domain} ({result.domain_confidence:.0%})")
        table.add_row("Topics", ", ".join(t.topic for t in result.top_topics[:5]))
        table.add_row("Patterns", str(len(result.patterns)))
        table.add_row("Task Types", ", ".join(result.task_types_detected))
        table.add_row("Total Entries", str(result.total_entries))
        table.add_row("Saved To", str(profile_path))
        console.print(table)

    elif action == "show":
        loaded = load_profile(profile_path)
        if loaded.domain == "unknown" and not loaded.top_topics:
            console.print("[yellow]No profile built yet. Run:[/yellow] fabrik profile build")
            return

        table = Table(title=f"Personal Profile — {loaded.domain.replace('_', ' ').title()}")
        table.add_column("Field", style="bold cyan")
        table.add_column("Value", style="green")
        table.add_row("Domain", f"{loaded.domain} ({loaded.domain_confidence:.0%})")
        table.add_row("Built", loaded.built_at or "unknown")
        table.add_row("Total Entries", str(loaded.total_entries))

        if loaded.top_topics:
            topics_str = "\n".join(f"  {t.topic}: {t.weight:.0%}" for t in loaded.top_topics[:8])
            table.add_row("Top Topics", topics_str)

        if loaded.patterns:
            table.add_row("Patterns", "\n".join(f"  {p}" for p in loaded.patterns))

        if loaded.task_types_detected:
            table.add_row("Task Types", ", ".join(loaded.task_types_detected))

        table.add_row(
            "Style", f"Formality: {loaded.style.formality:.0%}, Language: {loaded.style.language}"
        )
        console.print(table)

        console.print("\n[dim]System prompt preview:[/dim]")
        console.print(f"[italic]{loaded.to_system_prompt()}[/italic]")
    elif action == "build-incremental":
        console.print(Panel.fit("[bold]Incremental Profile Build...[/bold]"))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Analyzing new entries...", total=None)
            builder = ProfileBuilder(datalake_path=settings.datalake_path)
            result = builder.build_incremental(output_path=profile_path)

        if result.build_mode == "incremental":
            table = Table(title="Incremental Build")
            table.add_column("Field", style="bold cyan")
            table.add_column("Value", style="green")
            table.add_row("Total Entries", str(result.total_entries))
            table.add_row("Build Mode", result.build_mode)
            table.add_row("Topics", ", ".join(t.topic for t in result.top_topics[:5]))
            table.add_row("Drift History", str(len(result.drift_history)) + " events")
            if result.drift_history:
                latest = result.drift_history[-1]
                table.add_row(
                    "Latest Drift",
                    f"{latest['date']}: {', '.join(latest['topics_drifted'])} "
                    f"(mag: {latest['magnitude']})",
                )
            table.add_row("Saved To", str(profile_path))
            console.print(table)
        else:
            console.print("[yellow]No prior build found. Performed full build.[/yellow]")
            table = Table(title="Profile Built (Full)")
            table.add_column("Field", style="bold cyan")
            table.add_column("Value", style="green")
            table.add_row("Domain", f"{result.domain} ({result.domain_confidence:.0%})")
            table.add_row("Topics", ", ".join(t.topic for t in result.top_topics[:5]))
            table.add_row("Total Entries", str(result.total_entries))
            table.add_row("Saved To", str(profile_path))
            console.print(table)

    elif action == "drift":
        loaded = load_profile(profile_path)

        if not loaded.drift_history:
            console.print(
                "[yellow]No drift history yet. Run:[/yellow] fabrik profile build-incremental"
            )
            return

        table = Table(title="Profile Drift History")
        table.add_column("Date", style="cyan")
        table.add_column("Topics Drifted", style="yellow")
        table.add_column("Magnitude", style="red")

        for entry in loaded.drift_history:
            table.add_row(
                entry.get("date", "?"),
                ", ".join(entry.get("topics_drifted", [])),
                f"{entry.get('magnitude', 0):.4f}",
            )
        console.print(table)

    else:
        console.print("[yellow]Usage:[/yellow] fabrik profile [show|build|build-incremental|drift]")


@app.command()
def competence(
    action: str = typer.Argument("show", help="Action: show, build"),
):
    """Manage your competence map — measures depth of knowledge per topic."""
    from src.config import settings
    from src.core.competence_model import (
        CompetenceBuilder,
        load_competence_map,
    )

    competence_path = settings.data_dir / "profile" / "competence_map.json"

    if action == "build":
        console.print(Panel.fit("[bold]Building Competence Map...[/bold]"))

        # Try to load graph engine (optional)
        graph = None
        try:
            from src.knowledge.graph_engine import GraphEngine

            graph = GraphEngine()
            if not graph.load():
                graph = None
        except Exception:
            pass

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Analyzing competence...", total=None)
            builder = CompetenceBuilder(
                datalake_path=settings.datalake_path,
                graph_engine=graph,
            )
            result = builder.build(output_path=competence_path)

        if not result.topics:
            console.print("[yellow]No competence data found. Build your profile first.[/yellow]")
            return

        table = Table(title="Competence Map Built")
        table.add_column("Topic", style="bold cyan")
        table.add_column("Score", style="green", justify="right")
        table.add_column("Level", style="yellow")
        table.add_column("Entries", justify="right")
        table.add_column("Density", justify="right")
        table.add_column("Recency", justify="right")

        for entry in result.topics:
            table.add_row(
                entry.topic,
                f"{entry.score:.4f}",
                entry.level,
                str(entry.entries),
                f"{entry.entity_density:.4f}",
                f"{entry.recency_weight:.4f}",
            )

        console.print(table)
        console.print(f"\n[dim]Saved to: {competence_path}[/dim]")

        # Generate strategy overrides from outcome data
        from src.core.strategy_optimizer import MABStrategyOptimizer

        mab = MABStrategyOptimizer(settings.data_dir)
        overrides_path = settings.data_dir / "profile" / "strategy_overrides.json"
        override_count = mab.save_overrides(overrides_path)
        if override_count > 0:
            console.print(f"[bold]Strategy overrides:[/bold] {override_count} generated (MAB)")
        else:
            console.print("[dim]No strategy overrides needed (MAB defaults are fine)[/dim]")

        # Build learned classifier corpus from outcomes (FC-47)
        from src.core.task_router import LearnedClassifier

        corpus_path = settings.data_dir / "profile" / "router_corpus.json"
        learned = LearnedClassifier(corpus_path)
        corpus_size = learned.build_corpus(settings.datalake_path)
        if corpus_size > 0:
            fitted = learned.fit()
            if fitted:
                console.print(f"[bold]Learned classifier:[/bold] {corpus_size} entries, fitted")
            else:
                console.print(
                    f"[dim]Learned classifier: {corpus_size} entries "
                    f"(need {LearnedClassifier.MIN_CORPUS_SIZE} to activate)[/dim]"
                )
        else:
            console.print("[dim]Learned classifier: no accepted outcomes yet[/dim]")

    elif action == "show":
        loaded = load_competence_map(competence_path)
        if not loaded.topics:
            console.print(
                "[yellow]No competence map built yet. Run:[/yellow] fabrik competence build"
            )
            return

        table = Table(title="Competence Map")
        table.add_column("Topic", style="bold cyan")
        table.add_column("Score", style="green", justify="right")
        table.add_column("Level", style="yellow")
        table.add_column("Entries", justify="right")
        table.add_column("Last Activity")

        for entry in loaded.topics:
            table.add_row(
                entry.topic,
                f"{entry.score:.4f}",
                entry.level,
                str(entry.entries),
                entry.last_activity or "—",
            )

        console.print(table)
        console.print(f"\n[dim]Built at: {loaded.built_at}[/dim]")

        fragment = loaded.to_system_prompt_fragment()
        if fragment:
            console.print("\n[dim]System prompt fragment:[/dim]")
            console.print(f"[italic]{fragment}[/italic]")

    else:
        console.print("[yellow]Usage:[/yellow] fabrik competence [show|build]")


@app.command()
def outcomes(
    action: str = typer.Argument("stats", help="Action: show or stats"),
    topic: str | None = typer.Option(None, "--topic", "-t", help="Filter by topic"),
    task_type: str | None = typer.Option(None, "--task-type", help="Filter by task type"),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of records"),
):
    """View outcome tracking data."""
    import json as json_mod

    from src.config import settings

    outcomes_dir = settings.datalake_path / "01-raw" / "outcomes"

    # Read all outcome records
    records: list[dict] = []
    if outcomes_dir.exists():
        for filepath in sorted(outcomes_dir.glob("*_outcomes.jsonl")):
            for line in filepath.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json_mod.loads(line))
                except json_mod.JSONDecodeError:
                    continue

    # Apply filters
    if topic:
        records = [r for r in records if r.get("topic") == topic]
    if task_type:
        records = [r for r in records if r.get("task_type") == task_type]

    if action == "show":
        if not records:
            console.print("[yellow]No outcome records found.[/yellow]")
            return

        display = records[-limit:]

        table = Table(title="Outcome Records")
        table.add_column("Time", style="dim")
        table.add_column("Task Type", style="cyan")
        table.add_column("Topic")
        table.add_column("Outcome")
        table.add_column("Reason", style="dim")

        for r in display:
            ts = r.get("timestamp", "")[:19]
            outcome = r.get("outcome", "neutral")
            if outcome == "accepted":
                outcome_style = f"[green]{outcome}[/green]"
            elif outcome == "rejected":
                outcome_style = f"[red]{outcome}[/red]"
            else:
                outcome_style = f"[dim]{outcome}[/dim]"

            table.add_row(
                ts,
                r.get("task_type", "—"),
                r.get("topic") or "—",
                outcome_style,
                r.get("inference_reason", "")[:50],
            )

        console.print(table)
        console.print(f"[dim]Showing {len(display)} of {len(records)} records[/dim]")

    elif action == "stats":
        if not records:
            console.print("[yellow]No outcome records found.[/yellow]")
            return

        # Aggregate by (task_type, topic)
        buckets: dict[tuple[str, str], dict] = {}
        for r in records:
            key = (r.get("task_type", "general"), r.get("topic") or "—")
            if key not in buckets:
                buckets[key] = {"total": 0, "accepted": 0}
            outcome = r.get("outcome", "neutral")
            if outcome != "neutral":
                buckets[key]["total"] += 1
                if outcome == "accepted":
                    buckets[key]["accepted"] += 1

        table = Table(title="Outcome Stats")
        table.add_column("Task Type", style="cyan")
        table.add_column("Topic")
        table.add_column("Total", justify="right")
        table.add_column("Accepted", justify="right", style="green")
        table.add_column("Rate", justify="right")

        overall_total = 0
        overall_accepted = 0

        for (tt, tp), agg in sorted(buckets.items()):
            if agg["total"] == 0:
                continue
            rate = agg["accepted"] / agg["total"]
            table.add_row(
                tt,
                tp,
                str(agg["total"]),
                str(agg["accepted"]),
                f"{rate:.0%}",
            )
            overall_total += agg["total"]
            overall_accepted += agg["accepted"]

        if overall_total > 0:
            overall_rate = overall_accepted / overall_total
            table.add_row(
                "[bold]TOTAL[/bold]",
                "",
                f"[bold]{overall_total}[/bold]",
                f"[bold]{overall_accepted}[/bold]",
                f"[bold]{overall_rate:.0%}[/bold]",
            )

        console.print(table)

        # Show strategy overrides count
        overrides_path = settings.data_dir / "profile" / "strategy_overrides.json"
        if overrides_path.exists():
            try:
                overrides = json_mod.loads(overrides_path.read_text(encoding="utf-8"))
                console.print(f"[dim]Strategy overrides: {len(overrides)} active[/dim]")
            except (json_mod.JSONDecodeError, OSError):
                pass

    else:
        console.print(
            "[yellow]Usage:[/yellow] fabrik outcomes [show|stats] [--topic X] [--task-type X]"
        )


@app.command()
def router(
    action: str = typer.Argument("test", help="Action: test"),
    query: str = typer.Option(..., "--query", "-q", help="Query to classify"),
):
    """Test the adaptive task router — inspect classification without executing."""
    from src.config import settings
    from src.core.competence_model import get_active_competence_map
    from src.core.personal_profile import get_active_profile
    from src.core.task_router import TaskRouter, load_learned_classifier

    active_profile = get_active_profile()
    competence_map = get_active_competence_map()
    learned = load_learned_classifier(settings)

    task_router = TaskRouter(competence_map, active_profile, settings, learned_classifier=learned)

    async def run():
        decision = await task_router.route(query)

        table = Table(title="Routing Decision")
        table.add_column("Field", style="bold cyan")
        table.add_column("Value", style="green")
        table.add_row("Query", query)
        table.add_row("Task Type", f"{decision.task_type} ({decision.classification_method})")
        table.add_row(
            "Topic",
            (
                f"{decision.topic} ({decision.competence_level}, score={competence_map.get_score(decision.topic):.2f})"
                if decision.topic
                else "— (Unknown)"
            ),
        )
        table.add_row("Model", decision.model)
        table.add_row(
            "Strategy",
            f"graph_depth={decision.strategy.graph_depth}, "
            f"vector={decision.strategy.vector_weight}, "
            f"graph={decision.strategy.graph_weight}",
        )
        console.print(table)

        console.print("\n[dim]System prompt:[/dim]")
        console.print(f"[italic]{decision.system_prompt}[/italic]")

    if action == "test":
        async_run(run())
    else:
        console.print("[yellow]Usage:[/yellow] fabrik router test -q 'your query'")


@app.callback()
def main():
    """Fabrik-Codek: Claude Code's little brother 🤖"""
    pass


if __name__ == "__main__":
    app()
