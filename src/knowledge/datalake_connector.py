"""Connector for existing datalakes - reads accumulated knowledge."""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Literal

import aiofiles
import structlog

from src.config import settings

logger = structlog.get_logger()


@dataclass
class DatalakeFile:
    """Represents a file from the datalake."""

    path: Path
    relative_path: str
    datalake: str
    file_type: str
    size: int
    modified: datetime
    content: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def category(self) -> str:
        """Categorize file by its location/type."""
        rel = self.relative_path.lower()
        if "decision" in rel:
            return "decision"
        elif "learning" in rel:
            return "learning"
        elif "problem" in rel or "solution" in rel:
            return "problem_solution"
        elif "session" in rel or "conversation" in rel:
            return "conversation"
        elif "code-change" in rel:
            return "code_change"
        elif "training" in rel:
            return "training_data"
        elif "claude.md" in rel.lower():
            return "prompt_template"
        elif "readme" in rel.lower():
            return "documentation"
        elif rel.endswith(".py"):
            return "code_python"
        elif rel.endswith((".ts", ".tsx", ".js")):
            return "code_typescript"
        else:
            return "other"


class DatalakeConnector:
    """Connects to existing datalakes and extracts knowledge."""

    # File types to index
    INDEXABLE_EXTENSIONS = {
        ".md",
        ".txt",
        ".json",
        ".jsonl",
        ".py",
        ".ts",
        ".tsx",
        ".yaml",
        ".yml",
        ".toml",
    }

    # Directories to skip
    SKIP_DIRS = {
        "__pycache__",
        "node_modules",
        ".git",
        ".venv",
        "venv",
        ".next",
        "dist",
        "build",
        ".cache",
        "$RECYCLE.BIN",
        "System Volume Information",
    }

    # Max file size to read (5MB)
    MAX_FILE_SIZE = 5 * 1024 * 1024

    def __init__(self, path: Path | None = None):
        self.path = path or settings.datalake_path
        self._file_cache: dict[str, DatalakeFile] = {}

    async def scan_all(self) -> AsyncIterator[DatalakeFile]:
        """Scan the datalake directory."""
        if not self.path.exists():
            logger.warning("datalake_not_found", path=str(self.path))
            return

        datalake_name = self.path.name
        logger.info("scanning_datalake", name=datalake_name, path=str(self.path))

        async for file in self._scan_directory(self.path, datalake_name):
            yield file

    async def _scan_directory(
        self, root: Path, datalake_name: str
    ) -> AsyncIterator[DatalakeFile]:
        """Recursively scan a directory for indexable files."""

        def _walk_sync():
            """Synchronous walk for use in executor."""
            files = []
            for path in root.rglob("*"):
                if path.is_file():
                    # Skip if in excluded directory
                    if any(skip in path.parts for skip in self.SKIP_DIRS):
                        continue
                    # Check extension
                    if path.suffix.lower() in self.INDEXABLE_EXTENSIONS:
                        files.append(path)
            return files

        loop = asyncio.get_event_loop()
        files = await loop.run_in_executor(None, _walk_sync)

        for file_path in files:
            try:
                stat = file_path.stat()

                # Skip large files
                if stat.st_size > self.MAX_FILE_SIZE:
                    continue

                yield DatalakeFile(
                    path=file_path,
                    relative_path=str(file_path.relative_to(root)),
                    datalake=datalake_name,
                    file_type=file_path.suffix.lower(),
                    size=stat.st_size,
                    modified=datetime.fromtimestamp(stat.st_mtime),
                )
            except (OSError, ValueError) as e:
                logger.error("file_scan_error", path=str(file_path), error=str(e))

    async def read_file(self, file: DatalakeFile) -> DatalakeFile:
        """Read content of a datalake file."""
        try:
            async with aiofiles.open(file.path, "r", encoding="utf-8", errors="ignore") as f:
                content = await f.read()

            # Parse structured files
            if file.file_type == ".json":
                try:
                    file.metadata = json.loads(content)
                    file.content = content
                except json.JSONDecodeError:
                    file.content = content
            elif file.file_type == ".jsonl":
                lines = []
                for line in content.strip().split("\n"):
                    if line:
                        try:
                            lines.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                file.metadata = {"records": lines, "count": len(lines)}
                file.content = content
            else:
                file.content = content

            return file

        except (OSError, UnicodeDecodeError) as e:
            logger.error("file_read_error", path=str(file.path), error=str(e))
            file.content = ""
            return file

    async def get_decisions(self) -> list[DatalakeFile]:
        """Get all decision documents."""
        decisions = []
        async for file in self.scan_all():
            if file.category == "decision":
                file = await self.read_file(file)
                decisions.append(file)
        return decisions

    async def get_learnings(self) -> list[DatalakeFile]:
        """Get all learning documents."""
        learnings = []
        async for file in self.scan_all():
            if file.category == "learning":
                file = await self.read_file(file)
                learnings.append(file)
        return learnings

    async def get_prompt_templates(self) -> list[DatalakeFile]:
        """Get all CLAUDE.md and prompt templates."""
        templates = []
        async for file in self.scan_all():
            if file.category == "prompt_template":
                file = await self.read_file(file)
                templates.append(file)
        return templates

    async def get_conversations(self, limit: int = 100) -> list[DatalakeFile]:
        """Get conversation/session logs."""
        conversations = []
        async for file in self.scan_all():
            if file.category == "conversation":
                file = await self.read_file(file)
                conversations.append(file)
                if len(conversations) >= limit:
                    break
        return conversations

    async def get_stats(self) -> dict:
        """Get statistics about available data."""
        stats = {
            "total_files": 0,
            "by_datalake": {},
            "by_category": {},
            "by_type": {},
            "total_size_mb": 0,
        }

        async for file in self.scan_all():
            stats["total_files"] += 1
            stats["total_size_mb"] += file.size / (1024 * 1024)

            # By datalake
            if file.datalake not in stats["by_datalake"]:
                stats["by_datalake"][file.datalake] = 0
            stats["by_datalake"][file.datalake] += 1

            # By category
            if file.category not in stats["by_category"]:
                stats["by_category"][file.category] = 0
            stats["by_category"][file.category] += 1

            # By type
            if file.file_type not in stats["by_type"]:
                stats["by_type"][file.file_type] = 0
            stats["by_type"][file.file_type] += 1

        stats["total_size_mb"] = round(stats["total_size_mb"], 2)
        return stats

    async def search_files(
        self,
        query: str,
        category: str | None = None,
        datalake: str | None = None,
        limit: int = 50,
    ) -> list[DatalakeFile]:
        """Search files by content or path."""
        results = []
        query_lower = query.lower()

        async for file in self.scan_all():
            # Filter by category
            if category and file.category != category:
                continue

            # Filter by datalake
            if datalake and file.datalake != datalake:
                continue

            # Check path match
            if query_lower in file.relative_path.lower():
                file = await self.read_file(file)
                results.append(file)
                if len(results) >= limit:
                    break
                continue

            # Check content match (read file first)
            file = await self.read_file(file)
            if file.content and query_lower in file.content.lower():
                results.append(file)
                if len(results) >= limit:
                    break

        return results
