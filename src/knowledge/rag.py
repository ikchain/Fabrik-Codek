"""RAG System - Retrieval Augmented Generation for fabrik-codek.

Este módulo permite a fabrik-codek consultar el conocimiento acumulado
en el datalake para dar respuestas contextualizadas.

Uso:
    from src.knowledge.rag import RAGEngine

    rag = RAGEngine()
    await rag.index_datalake()  # Indexar documentos

    context = await rag.retrieve("cómo hacer autenticación JWT")
    response = await llm.generate(prompt, context=context)
"""

import asyncio
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import lancedb
from lancedb.pydantic import LanceModel, Vector

from src.config import settings

# Configurable via FABRIK_EMBEDDING_DIM env var (default: 768 for nomic-embed-text)
EMBEDDING_DIM = settings.embedding_dim


class Document(LanceModel):
    """Documento indexado en la base vectorial."""
    id: str
    text: str
    vector: Vector(EMBEDDING_DIM)
    source: str  # Archivo origen
    category: str  # Tipo de documento
    project: str  # Proyecto relacionado
    created_at: str


class RAGEngine:
    """Motor RAG para búsqueda semántica en el datalake."""

    def __init__(
        self,
        db_path: Path | None = None,
        embedding_model: str | None = None,
        ollama_host: str | None = None,
    ):
        self.db_path = db_path or settings.data_dir / "vectordb"
        self.embedding_model = embedding_model or settings.embedding_model
        self.ollama_host = ollama_host or settings.ollama_host
        self.table_name = "knowledge"

        self._db: lancedb.DBConnection | None = None
        self._table: lancedb.table.Table | None = None
        self._http_client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        await self._init()
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def _init(self):
        """Initialize database and HTTP client."""
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self.db_path))
        self._http_client = httpx.AsyncClient(timeout=60.0)

        # Create or open table
        if self.table_name in self._db.table_names():
            self._table = self._db.open_table(self.table_name)
        else:
            # Create empty table with schema
            self._table = self._db.create_table(
                self.table_name,
                schema=Document,
                mode="overwrite",
            )

    async def close(self):
        """Close connections."""
        if self._http_client:
            await self._http_client.aclose()

    async def _get_embedding(self, text: str) -> list[float]:
        """Get embedding from Ollama."""
        response = await self._http_client.post(
            f"{self.ollama_host}/api/embeddings",
            json={
                "model": self.embedding_model,
                "prompt": text,
            },
        )
        response.raise_for_status()
        return response.json()["embedding"]

    async def _get_embeddings_batch(self, texts: list[str], batch_size: int = 10) -> list[list[float]]:
        """Get embeddings for multiple texts."""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await asyncio.gather(
                *[self._get_embedding(text) for text in batch]
            )
            embeddings.extend(batch_embeddings)
        return embeddings

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind(". ")
                last_newline = chunk.rfind("\n")
                break_point = max(last_period, last_newline)
                if break_point > chunk_size // 2:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1

            chunks.append(chunk.strip())
            start = end - overlap

        return chunks

    async def index_file(self, file_path: Path, category: str = "general", project: str = "") -> int:
        """Index a single file."""
        if not file_path.exists():
            return 0

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):
            return 0

        if not content.strip():
            return 0

        # Chunk the content
        chunks = self._chunk_text(content)
        if not chunks:
            return 0

        # Generate embeddings
        embeddings = await self._get_embeddings_batch(chunks)

        # Create documents
        documents = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc_id = hashlib.md5(f"{file_path}:{i}:{chunk[:100]}".encode()).hexdigest()[:16]
            documents.append(Document(
                id=doc_id,
                text=chunk,
                vector=embedding,
                source=str(file_path),
                category=category,
                project=project,
                created_at=datetime.now().isoformat(),
            ))

        # Add to table
        self._table.add([doc.model_dump() for doc in documents])
        return len(documents)

    async def index_jsonl(self, file_path: Path, text_field: str = "output", category: str = "training") -> int:
        """Index a JSONL file (like training pairs)."""
        if not file_path.exists():
            return 0

        documents = []
        texts = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get(text_field, "")
                    if not text or len(text) < 50:
                        continue

                    # Combine instruction + output for better context
                    instruction = data.get("instruction", "")
                    if instruction:
                        text = f"Q: {instruction}\nA: {text}"

                    doc_id = data.get("id", hashlib.md5(text[:100].encode()).hexdigest()[:16])
                    project = data.get("source_file", "").split("/")[0] if "/" in data.get("source_file", "") else ""

                    documents.append({
                        "id": doc_id,
                        "text": text,
                        "source": str(file_path),
                        "category": data.get("category", category),
                        "project": project,
                        "created_at": datetime.now().isoformat(),
                    })
                    texts.append(text)
                except json.JSONDecodeError:
                    continue

        if not documents:
            return 0

        # Get embeddings
        embeddings = await self._get_embeddings_batch(texts)

        # Add vectors
        for doc, emb in zip(documents, embeddings):
            doc["vector"] = emb

        self._table.add(documents)
        return len(documents)

    async def index_datalake(self, force_reindex: bool = False) -> dict:
        """Index all relevant files from the datalake."""
        stats = {
            "files_indexed": 0,
            "chunks_created": 0,
            "errors": 0,
        }

        datalake = settings.datalake_path
        if not datalake.exists():
            return stats

        # Index training pairs (most valuable)
        training_dir = datalake / "02-processed" / "training-pairs"
        if training_dir.exists():
            for jsonl_file in training_dir.glob("*.jsonl"):
                try:
                    count = await self.index_jsonl(jsonl_file, category="training")
                    stats["chunks_created"] += count
                    stats["files_indexed"] += 1
                except (OSError, httpx.HTTPError, json.JSONDecodeError) as e:
                    stats["errors"] += 1

        # Index decisions
        decisions_dir = datalake / "03-metadata" / "decisions"
        if decisions_dir.exists():
            for md_file in decisions_dir.glob("**/*.md"):
                try:
                    count = await self.index_file(md_file, category="decision")
                    stats["chunks_created"] += count
                    stats["files_indexed"] += 1
                except (OSError, httpx.HTTPError, UnicodeDecodeError):
                    stats["errors"] += 1

        # Index learnings
        learnings_dir = datalake / "03-metadata" / "learnings"
        if learnings_dir.exists():
            for md_file in learnings_dir.glob("**/*.md"):
                try:
                    count = await self.index_file(md_file, category="learning")
                    stats["chunks_created"] += count
                    stats["files_indexed"] += 1
                except (OSError, httpx.HTTPError, UnicodeDecodeError):
                    stats["errors"] += 1

        # Index code changes (recent ones)
        code_changes_dir = datalake / "01-raw" / "code-changes"
        if code_changes_dir.exists():
            for jsonl_file in sorted(code_changes_dir.glob("*.jsonl"))[-30:]:  # Last 30 days
                try:
                    count = await self.index_jsonl(jsonl_file, text_field="description", category="code_change")
                    stats["chunks_created"] += count
                    stats["files_indexed"] += 1
                except (OSError, httpx.HTTPError, json.JSONDecodeError):
                    stats["errors"] += 1

        return stats

    async def retrieve(
        self,
        query: str,
        limit: int = 5,
        category: Optional[str] = None,
    ) -> list[dict]:
        """Retrieve relevant documents for a query."""
        query_embedding = await self._get_embedding(query)

        # Search
        results = self._table.search(query_embedding).limit(limit)

        if category:
            results = results.where(f"category = '{category}'")

        results = results.to_list()

        return [
            {
                "text": r["text"],
                "source": r["source"],
                "category": r["category"],
                "score": r.get("_distance", 0),
            }
            for r in results
        ]

    async def query_with_context(
        self,
        query: str,
        limit: int = 3,
    ) -> str:
        """Get query with injected context from RAG."""
        results = await self.retrieve(query, limit=limit)

        if not results:
            return query

        context_parts = []
        for r in results:
            context_parts.append(f"[{r['category']}] {r['text'][:500]}")

        context = "\n---\n".join(context_parts)

        return f"""Contexto relevante de tu base de conocimiento:
{context}

---
Pregunta: {query}

Responde usando el contexto anterior cuando sea relevante."""

    def get_stats(self) -> dict:
        """Get index statistics."""
        if not self._table:
            return {"total_documents": 0}

        try:
            count = self._table.count_rows()
            return {
                "total_documents": count,
                "db_path": str(self.db_path),
            }
        except (AttributeError, OSError):
            return {"total_documents": 0}


# Singleton instance
_rag_engine: RAGEngine | None = None


async def get_rag_engine() -> RAGEngine:
    """Get or create the global RAG engine."""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
        await _rag_engine._init()
    return _rag_engine
