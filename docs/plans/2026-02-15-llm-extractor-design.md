# Activar LLM Extractor para Knowledge Graph

**Fecha:** 2026-02-15
**Enfoque:** Extraccion directa con prompt mejorado (Approach A)

## Contexto

El `LLMExtractor` en `src/knowledge/extraction/llm_extractor.py` es un stub que retorna listas vacias. El heuristico (`HeuristicExtractor`) solo reconoce 65+ tecnologias y 28+ patrones de su diccionario, perdiendo entidades y relaciones no catalogadas. El LLM extractor complementa al heuristico procesando the accumulated training pairs via qwen2.5-coder:7b (Ollama).

## Decisiones de diseño

### Prompting: text-only extraction
- Enviar solo `instruction + category + topic` al LLM
- **NO** enviar el campo `output` (codigo largo de 2000+ chars que confunde al modelo 7B y consume tokens innecesarios)
- System prompt separado para mejor respuesta de qwen2.5-coder:7b
- Temperature baja (0.1) para output determinista
- Lista explicita de tipos validos en el prompt para reducir alucinaciones

### Concurrencia: secuencial
- 1 request a la vez contra Ollama (qwen2.5-coder:7b en una GPU, requests paralelas degradan throughput)
- `asyncio.sleep(delay)` configurable entre requests (default 0.5s)
- Circuit breaker: 5 errores consecutivos → abortar batch

### Scope: solo training pairs
- El LLM extractor solo se aplica a training pairs en esta primera iteracion
- Las otras fuentes (8 decisions, 11 learnings, auto-captures) son pocas y el heuristico las cubre bien

## Arquitectura

### Prompt

```
SYSTEM_PROMPT = """You are a technical knowledge extractor. Given a technical instruction,
extract entities (technologies, patterns, concepts, strategies) and their relationships.
Return ONLY valid JSON. No explanation."""

EXTRACTION_PROMPT = """Extract entities and relationships from this technical text.

Category: {category}
Topic: {topic}
Text: {instruction}

Entity types: technology, pattern, concept, strategy, error_type
Relation types: uses, depends_on, part_of, alternative_to, related_to, fixes, learned_from

Return JSON:
{{"entities": [{{"name": "...", "type": "..."}}], "relations": [{{"source": "...", "target": "...", "type": "..."}}]}}"""
```

### Flujo extract_from_pair()

1. Construir prompt con `instruction + category + topic`
2. Llamar `LLMClient.generate()` con system prompt + temperature 0.1
3. Parsear con `_parse_llm_response()` (ya implementado, reforzar)
4. Todos los triples LLM salen con confidence 0.6 (vs 0.7-0.8 del heuristico)
5. Retornar triples validos

### Flujo extract_batch()

```
errores_consecutivos = 0
MAX_ERRORES_CONSECUTIVOS = 5

por cada pair:
    try:
        triples = await extract_from_pair(pair)
        errores_consecutivos = 0
    except (httpx.ConnectError, httpx.TimeoutException):
        errores_consecutivos += 1
        if errores_consecutivos >= MAX_ERRORES_CONSECUTIVOS:
            break
    except Exception:
        errores_consecutivos += 1
    await asyncio.sleep(delay)
```

### Integracion con Pipeline

```
pair leido del JSONL
    |
    +-> heuristic.extract_from_pair(pair)  ->  triples heuristicos (siempre)
    |
    +-> llm_extractor.extract_from_pair(pair)  ->  triples LLM (si use_llm=True)
    |
    +-> merge: heuristicos + LLM  ->  engine.ingest_triple() para cada uno
```

- Deduplicacion delegada a `GraphEngine.ingest_triple()` (ya mergea entidades duplicadas)
- Stats: nuevo campo `llm_triples_extracted`
- Availability check al inicio de `build()`: si Ollama no responde, degradacion graceful (solo heuristico)
- Solo se aplica a `_extract_from_jsonl()` (training pairs)

## Manejo de errores

### _parse_llm_response() reforzado

| Caso | Manejo |
|------|--------|
| JSON envuelto en markdown fences | Stripear ` ```json ``` ` antes de parsear |
| Tipo desconocido (`"library"`) | Fallback a TECHNOLOGY (entity) o RELATED_TO (relation) |
| Entidades vacias | Descartar triple silenciosamente |
| JSON truncado | Intentar cerrar con `]}` y parsear lo recuperable |
| Solo texto sin JSON | Retornar `[]` |

### Circuit breaker

- 5 errores consecutivos de conexion/timeout → abortar batch
- Se resetea con cada exito
- Errores de parseo NO cuentan (el LLM respondio, solo dio basura)

### Timeout

- 30s por request (override del default 120s de LLMClient)

## Testing

### Unitarios: _parse_llm_response()

- JSON valido → triples correctos
- Markdown fences → strips y parsea
- JSON truncado → recupera o `[]`
- Respuesta vacia → `[]`
- Tipo desconocido → fallback
- Source/target vacio → triple descartado
- Campos extra → ignora sin error
- Solo texto → `[]`

### Unitarios: extract_from_pair()

- Mock LLMClient.generate() con LLMResponse fixtures reales
- Verificar prompt contiene instruction + category + topic (no output)
- Verificar temperature baja
- Verificar confidence 0.6 en triples

### Unitarios: extract_batch()

- Procesamiento secuencial verificado
- Circuit breaker: 5 TimeoutException consecutivas → corta
- Circuit breaker reset: error + exito + error → no corta
- Errores intermitentes: continua con demas pairs

### Integracion

- Pipeline con use_llm=True + LLMClient mockeado
- Stats incluyen llm_triples_extracted
- Degradacion graceful: check_availability() False → solo heuristico

### NO automatizar

- Test contra Ollama real (manual)

## Archivos a modificar

| Archivo | Cambio |
|---------|--------|
| `src/knowledge/extraction/llm_extractor.py` | Implementar extract_from_pair(), extract_batch(), reforzar _parse_llm_response() |
| `src/knowledge/extraction/pipeline.py` | Integrar LLM en _extract_from_jsonl(), availability check, stats |
| `tests/test_extraction.py` | Tests unitarios + integracion |

## Archivos que NO se tocan

- `src/knowledge/graph_schema.py` - schema estable
- `src/knowledge/graph_engine.py` - ya soporta merge
- `src/knowledge/extraction/heuristic.py` - no cambia
- `src/core/llm_client.py` - interfaz suficiente
- `src/interfaces/cli.py` - flag --no-llm ya existe
