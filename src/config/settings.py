"""Configuration settings for Fabrik-Codek."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="FABRIK_",
        extra="ignore",
    )

    # Project paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data")

    # Datalake path (flywheel data storage)
    datalake_path: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "data"
    )

    # LLM Configuration
    ollama_host: str = "http://localhost:11434"
    default_model: str = "qwen2.5-coder:14b"
    fallback_model: str = "qwen2.5-coder:32b"
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768

    # Generation parameters
    temperature: float = 0.1
    max_tokens: int = 4096
    context_window: int = 32768

    # Confidence thresholds for task routing
    confidence_threshold: float = 0.7
    escalation_threshold: float = 0.5

    # Flywheel settings
    flywheel_enabled: bool = True
    flywheel_auto_capture: bool = True
    flywheel_batch_size: int = 100

    # Vector DB
    vector_db: Literal["chromadb", "lancedb"] = "lancedb"
    collection_name: str = "fabrik_knowledge"

    # API settings
    api_host: str = "127.0.0.1"
    api_port: int = 8420
    mcp_port: int = 8421
    api_key: str | None = None
    api_cors_origins: list[str] = Field(default=["*"])

    # Knowledge Graph
    graph_vector_weight: float = 0.6
    graph_graph_weight: float = 0.4
    graph_default_depth: int = 2
    graph_min_weight: float = 0.3
    graph_decay_half_life_days: float = 90.0

    # Full-text search (Meilisearch) - optional
    meilisearch_url: str = "http://localhost:7700"
    meilisearch_key: str | None = None
    meilisearch_index: str = "fabrik_knowledge"
    fulltext_weight: float = 0.0  # 0.0 = disabled in RRF fusion

    # Context Gate
    context_gate_enabled: bool = True

    # Relevance filter — minimum query-text token overlap to keep RAG results
    relevance_threshold: float = 0.12

    # Logging
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "console"

    def get_data_path(self, subdir: str) -> Path:
        """Get path within data directory."""
        path = self.data_dir / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def raw_data_path(self) -> Path:
        return self.get_data_path("raw")

    @property
    def processed_data_path(self) -> Path:
        return self.get_data_path("processed")

    @property
    def embeddings_path(self) -> Path:
        return self.get_data_path("embeddings")

    @property
    def graph_db_path(self) -> Path:
        return self.get_data_path("graphdb")


settings = Settings()
