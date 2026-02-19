# Enriquecer Knowledge Graph con Reasoning de Session Transcripts

**Fecha:** 2026-02-15
**Enfoque:** Scan directo de thinking blocks (Approach A)

## Contexto

The accumulated session transcripts from Claude Code contain numerous thinking blocks con razonamiento tecnico profundo: decisiones de arquitectura, comparaciones de tecnologias, estrategias de debugging, patrones aprendidos. Actualmente NINGUNO de estos datos alimenta el knowledge graph.

El hook `extract_reasoning.py` captura reasoning en tiempo real (1 tool_use a la vez), pero no procesa el historico. Este paso hace un batch scan de todos los transcripts para extraer relaciones de alta calidad del reasoning acumulado.

## Decisiones de diseno

### Approach: Scan directo de thinking blocks
- Escanea cada transcript JSONL buscando mensajes `assistant` con bloques `thinking`
- Extrae el texto del thinking block y lo pasa por `HeuristicExtractor`
- NO reconstruye cadena parentUuid (demasiada complejidad para poco beneficio marginal)
- El hook ya cubre contexto tool_use para captures nuevos

### Scope: Configurable project filter
- Optional filter via `FABRIK_PROJECT_FILTER` env var
- Without filter, processes all project transcript directories
- Useful to focus on specific project transcripts

### Extraccion: Solo heuristica
- `HeuristicExtractor` ya tiene diccionarios de 65+ tecnologias, 28+ patrones, 14+ estrategias
- El heuristico es rapido, determinista y probado con 83+ tests
- Sin LLM: 21K blocks se procesan en <1 minuto vs ~3 horas con Ollama
- Confidence 0.65 para triples de thinking blocks (mayor que auto-captures: 0.4)

### Filtro de calidad: >100 chars
- Thinking blocks <100 chars no tienen razonamiento util
- Misma regla que `determine_confidence()` del script existente

## Arquitectura

### TranscriptExtractor

**Archivo:** `src/knowledge/extraction/transcript_extractor.py`

```python
class TranscriptExtractor:
    """Extract knowledge triples from Claude Code session transcript thinking blocks."""

    def __init__(self):
        self.heuristic = HeuristicExtractor()

    def extract_from_transcript(self, transcript_path: Path) -> list[Triple]:
        """Parse transcript JSONL, extract thinking blocks, return triples."""

    def scan_all_transcripts(self, transcripts_dir: Path, project_filter: str | None = None) -> tuple[list[Triple], dict]:
        """Scan project transcripts. Returns (triples, stats)."""
```

#### Flujo extract_from_transcript()

1. Abrir JSONL linea por linea
2. Filtrar `type == "assistant"` con `message.content` de tipo lista
3. Por cada bloque con `type == "thinking"` y `len(thinking) > 100`:
   - Pasar texto por `HeuristicExtractor._find_technologies()`, `_extract_patterns()`, `_extract_strategies()`, `_extract_errors()`
   - Generar triples con confidence 0.65
   - `source_doc` = path relativo del transcript (para trazabilidad)
4. Generar co-occurrence relations entre tecnologias encontradas en el mismo thinking block

#### Flujo scan_all_transcripts()

1. Listar subdirectorios de `~/.claude/projects/`
2. Optionally filter by project name (FABRIK_PROJECT_FILTER)
3. Para cada directorio, procesar todos los `.jsonl`
4. Retornar (triples, stats)

### Integracion con Pipeline

**Archivo:** `src/knowledge/extraction/pipeline.py`

Nuevo paso 6 en `build()`:

```
paso 1: training-pairs (heuristico + LLM)
paso 2: decisions (heuristico)
paso 3: learnings (heuristico)
paso 4: auto-captures (heuristico)
paso 5: enriched captures (heuristico)
paso 6: session transcripts (heuristico)  <-- NUEVO
```

- Flag `include_transcripts: bool = False` en `__init__` y `build()`
- Transcripts se trackean en `processed_files` por path relativo + mtime (incremental)
- Nuevo campo en `_stats`: `transcript_triples_extracted`

### CLI

**Archivo:** `src/interfaces/cli.py`

```
fabrik graph build --include-transcripts
```

Sin el flag, el build se comporta exactamente como antes.

## Testing

### Unitarios: TranscriptExtractor

| Test | Que verifica |
|------|-------------|
| `test_extract_thinking_blocks` | Parseo basico de JSONL con thinking blocks |
| `test_filter_short_blocks` | Ignora thinking <100 chars |
| `test_extract_triples_from_thinking` | Extrae tech/patterns del texto |
| `test_confidence_level` | Triples tienen confidence 0.65 |
| `test_project_filter` | Only processes dirs matching project filter |
| `test_empty_transcript` | Archivo vacio retorna [] |
| `test_malformed_jsonl` | Lineas corruptas no crashean |

### Integracion: Pipeline con transcripts

| Test | Que verifica |
|------|-------------|
| `test_pipeline_with_transcripts` | build() con include_transcripts=True procesa transcripts |
| `test_pipeline_without_flag` | build() default no toca transcripts |
| `test_pipeline_transcript_stats` | Stats incluyen `transcript_triples_extracted` |
| `test_pipeline_incremental` | Segundo build no reprocesa transcripts sin cambios |

Todos con tmpdir y fixtures JSONL sinteticos.

## Archivos a modificar

| Archivo | Cambio |
|---------|--------|
| `src/knowledge/extraction/transcript_extractor.py` | **CREAR** - TranscriptExtractor |
| `src/knowledge/extraction/pipeline.py` | Integrar transcripts como paso 6 |
| `src/interfaces/cli.py` | Flag --include-transcripts |
| `tests/test_extraction.py` | Tests unitarios + integracion |

## Archivos que NO se tocan

- `scripts/extract_reasoning.py` - sigue como helper standalone del hook
- `src/knowledge/graph_schema.py` - schema estable
- `src/knowledge/graph_engine.py` - ya soporta merge
- `src/knowledge/extraction/heuristic.py` - no cambia
- `src/knowledge/extraction/llm_extractor.py` - no aplica
