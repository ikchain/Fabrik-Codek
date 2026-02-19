# Graph Completion por Inferencia Transitiva

**Fecha:** 2026-02-15
**Enfoque:** Metodo complete() en GraphEngine (Approach A)

## Contexto

El knowledge graph tiene 62,897+ triples extraidos de training pairs, auto-captures y session transcripts. Muchas relaciones implicitas no estan materializadas: si FastAPI DEPENDS_ON Starlette y Starlette DEPENDS_ON uvicorn, la relacion FastAPI->uvicorn no existe explicitamente. Graph completion infiere estas relaciones transitivas para densificar el grafo.

## Decisiones de diseno

### Relaciones transitivas: solo DEPENDS_ON y PART_OF
- DEPENDS_ON: A depends B, B depends C → A depends C (logicamente solida)
- PART_OF: A part_of B, B part_of C → A part_of C (composicion jerarquica)
- USES, RELATED_TO, FIXES, ALTERNATIVE_TO, LEARNED_FROM: NO transitivas (demasiado vagas o sin logica transitiva)

### Confidence: 0.3
- Minimo para ser visible en BFS default (min_weight=0.3)
- Distingue relaciones inferidas de directas (0.4-0.8)
- No domina el grafo pero aporta conectividad

### Un solo nivel de transitividad
- Solo infiere A->C de A->B->C, no cadenas mas largas
- Suficiente para nuestro dominio tecnico
- Sin riesgo de explosion combinatoria

### No sobreescribe relaciones existentes
- Si A->C ya existe como edge directo, no se modifica
- Solo crea edges nuevos

## Arquitectura

### GraphEngine.complete()

**Archivo:** `src/knowledge/graph_engine.py`

```python
TRANSITIVE_RELATIONS = [RelationType.DEPENDS_ON, RelationType.PART_OF]
INFERRED_CONFIDENCE = 0.3

def complete(self) -> dict:
    """Infer transitive relations for DEPENDS_ON and PART_OF."""
```

#### Flujo

1. Para cada RelationType transitiva (DEPENDS_ON, PART_OF):
   a. Recopilar todos los edges de ese tipo
   b. Agrupar por source y target para lookup rapido
   c. Para cada par (A->B, B->C) donde target de uno == source del otro:
      - Si NO existe edge A->C: crear con weight=0.3, source_docs=["inferred:transitive"], metadata={"inferred": true}
2. Retornar stats: {inferred_count, depends_on_inferred, part_of_inferred}

#### Propiedades de edges inferidos

| Campo | Valor |
|-------|-------|
| weight | 0.3 |
| source_docs | ["inferred:transitive"] |
| metadata | {"inferred": true} |
| relation_type | Mismo que la cadena |

### Integracion con Pipeline

**Archivo:** `src/knowledge/extraction/pipeline.py`

Paso 7 en `build()`, despues de transcripts, antes de `engine.save()`:

```python
# 7. Graph completion (transitive inference)
completion_stats = self.engine.complete()
self._stats["inferred_triples"] = completion_stats["inferred_count"]
```

Se ejecuta SIEMPRE (rapido, determinista, sin flag).
Nuevo campo en `_stats`: `inferred_triples`.

### CLI

**Archivo:** `src/interfaces/cli.py`

1. Mostrar `inferred_triples` en tabla de build si > 0
2. Nuevo action `fabrik graph complete` para ejecutar completion standalone

## Testing

### Unitarios: GraphEngine.complete()

| Test | Que verifica |
|------|-------------|
| `test_complete_depends_on` | A->B->C con DEPENDS_ON → infiere A->C |
| `test_complete_part_of` | A->B->C con PART_OF → infiere A->C |
| `test_complete_no_duplicate` | Si A->C existe, no crea duplicado |
| `test_complete_different_types` | A depends_on B, B part_of C → NO infiere |
| `test_complete_uses_not_transitive` | A uses B, B uses C → NO infiere |
| `test_complete_stats` | Stats retornan conteos correctos |
| `test_complete_inferred_metadata` | weight=0.3, metadata={"inferred": true} |
| `test_complete_empty_graph` | Grafo vacio → stats con 0s |

### Integracion: Pipeline con completion

| Test | Que verifica |
|------|-------------|
| `test_pipeline_runs_completion` | build() ejecuta completion, stats incluyen inferred_triples |

## Archivos a modificar

| Archivo | Cambio |
|---------|--------|
| `src/knowledge/graph_engine.py` | Nuevo metodo complete() |
| `src/knowledge/extraction/pipeline.py` | Paso 7 completion |
| `src/interfaces/cli.py` | Mostrar stats + action complete |
| `tests/test_extraction.py` | 9 tests (8 unit + 1 integracion) |

## Archivos que NO se tocan

- `src/knowledge/graph_schema.py` - schema estable
- `src/knowledge/extraction/heuristic.py` - no cambia
- `src/knowledge/extraction/llm_extractor.py` - no aplica
- `src/knowledge/extraction/transcript_extractor.py` - no aplica
- `src/knowledge/hybrid_rag.py` - no cambia
