"""Fabrik Logger - Centralized logging for Fabrik-Codek projects.

This is the single source of truth for data collection across all projects.
Import this logger in any project to capture interactions for the flywheel.

ZERO EXTERNAL DEPENDENCIES - Works standalone in any project.

Usage:
    from fabrik_codek.flywheel import get_logger

    logger = get_logger()
    logger.log_code_change(
        file_modified="src/main.py",
        change_type="updated",
        description="Added error handling",
        reasoning="Prevent crashes on invalid input",
        lesson_learned="Always validate user input at boundaries"
    )
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

# Centralized datalake path - override with FABRIK_DATALAKE_PATH env var
DATALAKE_PATH = Path(os.environ.get("FABRIK_DATALAKE_PATH", Path(__file__).parent.parent / "data"))

# Quality thresholds for fine-tuning data
# These are MANDATORY - data below these thresholds is REJECTED
QUALITY_THRESHOLDS = {
    "min_reasoning_length": 100,      # Reasoning must explain WHY
    "min_lesson_length": 50,          # Lessons must be actionable
    "min_description_length": 50,     # Descriptions must have context
    "min_how_fixed_length": 100,      # Fixes must explain the solution
}


class QualityValidationError(ValueError):
    """Raised when data doesn't meet quality standards for fine-tuning."""
    pass


class FabrikLogger:
    """Centralized logger for the fabrik-codek flywheel.

    All data goes to a single datalake with subdirectories:
    - 01-raw/sessions/
    - 01-raw/code-changes/
    - 01-raw/interactions/
    - 01-raw/errors/
    - 03-metadata/decisions/
    - 03-metadata/learnings/
    """

    def __init__(
        self,
        project_name: str = "",
        datalake_path: Path | None = None,
    ):
        self.project_name = project_name or self._detect_project()
        self.datalake_path = datalake_path or DATALAKE_PATH
        self.session_id = self._generate_session_id()

        # Ensure directories exist
        self._ensure_directories()

    def _detect_project(self) -> str:
        """Detect project name from git or cwd."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return Path(result.stdout.strip()).name
        except Exception:
            pass
        return Path.cwd().name

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        timestamp = datetime.now().isoformat()
        raw = f"{self.project_name}-{timestamp}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def _ensure_directories(self) -> None:
        """Ensure all datalake directories exist."""
        dirs = [
            self.datalake_path / "01-raw" / "sessions",
            self.datalake_path / "01-raw" / "code-changes",
            self.datalake_path / "01-raw" / "interactions",
            self.datalake_path / "01-raw" / "errors",
            self.datalake_path / "02-processed" / "training-pairs",
            self.datalake_path / "02-processed" / "embeddings",
            self.datalake_path / "02-processed" / "curated",
            self.datalake_path / "03-metadata" / "decisions",
            self.datalake_path / "03-metadata" / "learnings",
            self.datalake_path / "03-metadata" / "patterns",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def _get_filepath(self, category: str, subdir: str = "01-raw") -> Path:
        """Get filepath for today's log file."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.datalake_path / subdir / category / f"{today}_{category}.jsonl"

    def _write_entry(self, entry: dict, category: str, subdir: str = "01-raw") -> None:
        """Write entry to JSONL file (append)."""
        filepath = self._get_filepath(category, subdir)

        try:
            with open(filepath, "a", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, default=str)
                f.write("\n")
        except Exception as e:
            # Never fail silently but also never block
            print(f"  [fabrik-logger] Write error: {e}", file=sys.stderr)

    def _validate_quality(
        self,
        entry_type: str,
        reasoning: str = "",
        lesson_learned: str = "",
        description: str = "",
        how_fixed: str = "",
    ) -> None:
        """Validate data quality before saving. Raises QualityValidationError if invalid.

        This ensures we only collect HIGH-QUALITY data for fine-tuning.
        Low-quality data degrades model performance (causes hallucinations).
        """
        errors = []

        # Validate reasoning (required for decisions and code changes)
        if entry_type in ("decision", "code_change"):
            min_len = QUALITY_THRESHOLDS["min_reasoning_length"]
            if len(reasoning.strip()) < min_len:
                errors.append(
                    f"reasoning debe tener >= {min_len} chars (tiene {len(reasoning.strip())}). "
                    f"Explica el PORQUÉ de la decisión."
                )

        # Validate lesson_learned (required for all training-worthy entries)
        if entry_type in ("decision", "error", "code_change"):
            min_len = QUALITY_THRESHOLDS["min_lesson_length"]
            if len(lesson_learned.strip()) < min_len:
                errors.append(
                    f"lesson_learned debe tener >= {min_len} chars (tiene {len(lesson_learned.strip())}). "
                    f"¿Qué aprendimos para el futuro?"
                )

        # Validate how_fixed for errors
        if entry_type == "error":
            min_len = QUALITY_THRESHOLDS["min_how_fixed_length"]
            if len(how_fixed.strip()) < min_len:
                errors.append(
                    f"how_fixed debe tener >= {min_len} chars (tiene {len(how_fixed.strip())}). "
                    f"Explica paso a paso cómo se resolvió."
                )

        # Validate description
        if entry_type in ("code_change", "error"):
            min_len = QUALITY_THRESHOLDS["min_description_length"]
            actual = description or how_fixed  # error uses how_fixed as main description
            if len(actual.strip()) < min_len:
                errors.append(
                    f"description debe tener >= {min_len} chars. "
                    f"Proporciona contexto suficiente."
                )

        if errors:
            error_msg = (
                f"\n[fabrik-logger] DATOS RECHAZADOS - Calidad insuficiente:\n"
                + "\n".join(f"  ❌ {e}" for e in errors)
                + "\n\nLos datos de baja calidad causan alucinaciones en el modelo."
            )
            raise QualityValidationError(error_msg)

    # =========================================================================
    # Core logging methods
    # =========================================================================

    def log_session_start(
        self,
        context: str = "",
        tags: list[str] | None = None,
    ) -> None:
        """Log session start."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "session_start",
            "project": self.project_name,
            "session_id": self.session_id,
            "context": context,
            "tags": tags or ["session-start"],
        }
        self._write_entry(entry, "sessions")

    def log_code_change(
        self,
        file_modified: str,
        change_type: Literal["created", "updated", "deleted", "refactored"],
        description: str,
        reasoning: str = "",
        lesson_learned: str = "",
        commit_hash: str = "",
        tags: list[str] | None = None,
    ) -> None:
        """Log a code change with context.

        Raises:
            QualityValidationError: If reasoning or lesson_learned don't meet quality thresholds.
        """
        # Validate quality BEFORE saving
        self._validate_quality(
            entry_type="code_change",
            reasoning=reasoning,
            lesson_learned=lesson_learned,
            description=description,
        )

        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "code_change",
            "project": self.project_name,
            "session_id": self.session_id,
            "file_modified": file_modified,
            "change_type": change_type,
            "description": description,
            "reasoning": reasoning,
            "lesson_learned": lesson_learned,
            "commit_hash": commit_hash,
            "tags": tags or [],
        }
        self._write_entry(entry, "code-changes")

    def log_interaction(
        self,
        user_input: str,
        assistant_output: str,
        interaction_type: Literal[
            "prompt_response",
            "code_generation",
            "code_review",
            "refactor",
            "search",
            "documentation",
            "test",
            "other",
        ] = "prompt_response",
        context: str = "",
        reasoning: str = "",
        lesson_learned: str = "",
        model: str = "",
        tokens_used: int = 0,
        latency_ms: float = 0,
        feedback: Literal["positive", "negative", "neutral", "none"] = "none",
        tags: list[str] | None = None,
    ) -> None:
        """Log an LLM interaction."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "interaction",
            "interaction_type": interaction_type,
            "project": self.project_name,
            "session_id": self.session_id,
            "user_input": user_input,
            "assistant_output": assistant_output,
            "context": context,
            "reasoning": reasoning,
            "lesson_learned": lesson_learned,
            "model": model,
            "tokens_used": tokens_used,
            "latency_ms": latency_ms,
            "feedback": feedback,
            "tags": tags or [],
        }
        self._write_entry(entry, "interactions")

    def log_error(
        self,
        error_type: str,
        error_message: str,
        how_fixed: str,
        lesson_learned: str = "",
        stack_trace: str = "",
        affected_file: str = "",
        tags: list[str] | None = None,
    ) -> None:
        """Log an error and its resolution.

        Raises:
            QualityValidationError: If how_fixed or lesson_learned don't meet quality thresholds.
        """
        # Validate quality BEFORE saving
        self._validate_quality(
            entry_type="error",
            lesson_learned=lesson_learned,
            how_fixed=how_fixed,
        )

        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "error",
            "project": self.project_name,
            "session_id": self.session_id,
            "error_type": error_type,
            "error_message": error_message,
            "how_fixed": how_fixed,
            "lesson_learned": lesson_learned,
            "stack_trace": stack_trace,
            "affected_file": affected_file,
            "tags": tags or ["error", "troubleshooting"],
        }
        self._write_entry(entry, "errors")

    def log_decision(
        self,
        decision: str,
        alternatives_considered: list[str],
        chosen_option: str,
        reasoning: str,
        lesson_learned: str = "",
        affected_components: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Log a technical decision.

        Raises:
            QualityValidationError: If reasoning or lesson_learned don't meet quality thresholds.
        """
        # Validate quality BEFORE saving
        self._validate_quality(
            entry_type="decision",
            reasoning=reasoning,
            lesson_learned=lesson_learned,
        )

        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "decision",
            "project": self.project_name,
            "session_id": self.session_id,
            "decision": decision,
            "alternatives_considered": alternatives_considered,
            "chosen_option": chosen_option,
            "reasoning": reasoning,
            "lesson_learned": lesson_learned,
            "affected_components": affected_components or [],
            "tags": tags or ["decision", "architecture"],
        }
        self._write_entry(entry, "decisions", "03-metadata")

    def log_learning(
        self,
        topic: str,
        insight: str,
        context: str = "",
        source: str = "",
        applicable_to: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Log a learning or insight."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "learning",
            "project": self.project_name,
            "session_id": self.session_id,
            "topic": topic,
            "insight": insight,
            "context": context,
            "source": source,
            "applicable_to": applicable_to or [],
            "tags": tags or ["learning"],
        }
        self._write_entry(entry, "learnings", "03-metadata")

    def log_review_feedback(
        self,
        review_type: Literal["code_review", "pr_review", "commit_review"],
        suggestion: str,
        was_correct: bool,
        actual_finding: str = "",
        why_wrong: str = "",
        model_used: str = "qwen2.5-coder:7b",
        files_reviewed: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Log feedback about a Fabrik code review suggestion.

        This creates a feedback loop to identify when Fabrik hallucinates
        or makes incorrect suggestions, enabling targeted improvements.

        Args:
            review_type: Type of review (code_review, pr_review, commit_review)
            suggestion: The suggestion Fabrik made
            was_correct: True if suggestion was valid, False if hallucination/error
            actual_finding: What was actually found when verifying (if incorrect)
            why_wrong: Explanation of why the suggestion was wrong (if incorrect)
            model_used: Which Fabrik model version was used
            files_reviewed: List of files that were reviewed
            tags: Additional tags for categorization
        """
        # For incorrect suggestions, we need explanations
        if not was_correct:
            if len(actual_finding.strip()) < 50:
                raise QualityValidationError(
                    "actual_finding debe tener >= 50 chars para sugerencias incorrectas. "
                    "Describe qué encontraste realmente en el código."
                )
            if len(why_wrong.strip()) < 50:
                raise QualityValidationError(
                    "why_wrong debe tener >= 50 chars para sugerencias incorrectas. "
                    "Explica por qué la sugerencia de Fabrik estaba mal."
                )

        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "review_feedback",
            "review_type": review_type,
            "project": self.project_name,
            "session_id": self.session_id,
            "suggestion": suggestion,
            "was_correct": was_correct,
            "actual_finding": actual_finding,
            "why_wrong": why_wrong,
            "model_used": model_used,
            "files_reviewed": files_reviewed or [],
            "tags": tags or ["feedback", "fabrik-review"],
        }

        # Store in dedicated feedback directory
        feedback_dir = self.datalake_path / "03-metadata" / "review-feedback"
        feedback_dir.mkdir(parents=True, exist_ok=True)

        today = datetime.now().strftime("%Y-%m-%d")
        filepath = feedback_dir / f"{today}_review-feedback.jsonl"

        try:
            with open(filepath, "a", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, default=str)
                f.write("\n")
        except Exception as e:
            print(f"  [fabrik-logger] Write error: {e}", file=sys.stderr)

    def log_pattern(
        self,
        pattern_name: str,
        description: str,
        when_to_use: str,
        example_code: str = "",
        anti_patterns: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Log a reusable pattern."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "pattern",
            "project": self.project_name,
            "session_id": self.session_id,
            "pattern_name": pattern_name,
            "description": description,
            "when_to_use": when_to_use,
            "example_code": example_code,
            "anti_patterns": anti_patterns or [],
            "tags": tags or ["pattern"],
        }
        self._write_entry(entry, "patterns", "03-metadata")

    # =========================================================================
    # Domain-specific methods (extensible)
    # =========================================================================

    def log_pipeline_event(
        self,
        event_type: str,
        details: dict[str, Any],
        reasoning: str = "",
        lesson_learned: str = "",
        tags: list[str] | None = None,
    ) -> None:
        """Log a pipeline/processing event (for data processing projects)."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "pipeline_event",
            "event_type": event_type,
            "project": self.project_name,
            "session_id": self.session_id,
            "details": details,
            "reasoning": reasoning,
            "lesson_learned": lesson_learned,
            "tags": tags or ["pipeline"],
        }
        self._write_entry(entry, "interactions")

    def log_api_call(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        latency_ms: float,
        request_summary: str = "",
        response_summary: str = "",
        error: str = "",
        tags: list[str] | None = None,
    ) -> None:
        """Log an API call (useful for debugging integrations)."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "api_call",
            "project": self.project_name,
            "session_id": self.session_id,
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "latency_ms": latency_ms,
            "request_summary": request_summary,
            "response_summary": response_summary,
            "error": error,
            "tags": tags or ["api"],
        }
        self._write_entry(entry, "interactions")


# Singleton instance
_logger: FabrikLogger | None = None


def get_logger(project_name: str = "") -> FabrikLogger:
    """Get or create the global FabrikLogger instance."""
    global _logger
    if _logger is None:
        _logger = FabrikLogger(project_name=project_name)
    return _logger


def reset_logger() -> None:
    """Reset the global logger (useful for testing)."""
    global _logger
    _logger = None
