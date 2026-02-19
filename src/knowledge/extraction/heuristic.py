"""Heuristic-based entity and relation extraction from training pairs."""

import re

from src.knowledge.graph_schema import EntityType, RelationType, Triple

# --- Known dictionaries ---

KNOWN_TECHNOLOGIES: dict[str, list[str]] = {
    "fastapi": ["fast-api", "fast api"],
    "angular": ["ng"],
    "react": ["reactjs", "react.js"],
    "vue": ["vuejs", "vue.js"],
    "django": [],
    "flask": [],
    "express": ["expressjs", "express.js"],
    "nestjs": ["nest.js", "nest"],
    "nextjs": ["next.js", "next"],
    "postgresql": ["postgres", "psql", "pg"],
    "mongodb": ["mongo"],
    "redis": [],
    "elasticsearch": ["elastic", "es"],
    "docker": [],
    "kubernetes": ["k8s"],
    "terraform": ["tf"],
    "ansible": [],
    "github actions": ["gh actions"],
    "gitlab ci": ["gitlab-ci"],
    "jenkins": [],
    "pydantic": [],
    "sqlalchemy": ["sa"],
    "prisma": [],
    "typeorm": [],
    "rxjs": ["rx.js", "rx"],
    "ngrx": [],
    "langchain": [],
    "langgraph": [],
    "chromadb": ["chroma"],
    "lancedb": ["lance"],
    "pinecone": [],
    "ollama": [],
    "openai": ["gpt"],
    "anthropic": ["claude"],
    "huggingface": ["hf", "hugging face"],
    "pytorch": ["torch"],
    "tensorflow": ["tf"],
    "numpy": ["np"],
    "pandas": ["pd"],
    "celery": [],
    "rabbitmq": ["rmq"],
    "kafka": [],
    "nginx": [],
    "graphql": ["gql"],
    "grpc": [],
    "websockets": ["ws"],
    "starlette": [],
    "uvicorn": [],
    "pytest": [],
    "jest": [],
    "cypress": [],
    "playwright": [],
    "ruff": [],
    "eslint": [],
    "prettier": [],
}

KNOWN_PATTERNS: dict[str, list[str]] = {
    "hexagonal architecture": ["hexagonal", "ports and adapters"],
    "repository pattern": ["repository"],
    "cqrs": ["command query responsibility segregation"],
    "event sourcing": ["event-sourcing"],
    "saga pattern": ["saga"],
    "observer pattern": ["observer"],
    "strategy pattern": ["strategy"],
    "factory pattern": ["factory"],
    "singleton pattern": ["singleton"],
    "dependency injection": ["di", "injection"],
    "middleware pattern": ["middleware"],
    "decorator pattern": ["decorator"],
    "adapter pattern": ["adapter"],
    "facade pattern": ["facade"],
    "react agent": ["react pattern", "react agent pattern"],
    "chain of thought": ["cot", "chain-of-thought"],
    "rag": ["retrieval augmented generation"],
    "fine-tuning": ["finetuning", "fine tuning"],
    "prompt engineering": ["prompt design"],
    "few-shot": ["few shot", "few-shot learning"],
    "zero-shot": ["zero shot"],
    "lora": ["low-rank adaptation"],
    "qlora": [],
    "microservices": ["micro-services"],
    "monorepo": ["mono-repo"],
    "domain-driven design": ["ddd"],
    "test-driven development": ["tdd"],
    "behavior-driven development": ["bdd"],
    "clean architecture": ["clean arch"],
    "solid principles": ["solid"],
}

KNOWN_ERROR_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("connection error", re.compile(r"connection\s*(?:error|timeout|refused|reset)", re.I)),
    ("timeout error", re.compile(r"timeout\s*(?:error|exception)?|timed?\s*out", re.I)),
    ("memory error", re.compile(r"out\s*of\s*memory|oom|memory\s*error", re.I)),
    ("import error", re.compile(r"import\s*error|module\s*not\s*found", re.I)),
    ("type error", re.compile(r"type\s*error|type\s*mismatch", re.I)),
    ("null pointer", re.compile(r"null\s*(?:pointer|reference)|undefined\s*is\s*not", re.I)),
    ("race condition", re.compile(r"race\s*condition|deadlock|concurrent", re.I)),
    ("overfitting", re.compile(r"overfit(?:ting)?|over-fit", re.I)),
    ("underfitting", re.compile(r"underfit(?:ting)?|under-fit", re.I)),
    ("cors error", re.compile(r"cors\s*(?:error|issue|problem)", re.I)),
    ("authentication error", re.compile(r"auth(?:entication)?\s*(?:error|fail)", re.I)),
    ("permission error", re.compile(r"permission\s*(?:denied|error)|forbidden", re.I)),
    ("validation error", re.compile(r"validation\s*error|invalid\s*input", re.I)),
]

FILE_EXT_TO_TECH: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".js": "javascript",
    ".tsx": "react",
    ".jsx": "react",
    ".java": "java",
    ".go": "golang",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".sql": "postgresql",
    ".tf": "terraform",
    ".yml": "ansible",
    ".yaml": "ansible",
    ".dockerfile": "docker",
    ".sh": "bash",
    ".css": "css",
    ".scss": "sass",
    ".html": "html",
    ".vue": "vue",
    ".svelte": "svelte",
}

KNOWN_STRATEGIES: list[tuple[str, re.Pattern]] = [
    ("retry with backoff", re.compile(r"retry.*backoff|exponential\s*backoff|backoff.*retry", re.I)),
    ("early stopping", re.compile(r"early\s*stop(?:ping)?", re.I)),
    ("caching", re.compile(r"cach(?:e|ing)\s*(?:strategy|layer|response)", re.I)),
    ("lazy loading", re.compile(r"lazy\s*load(?:ing)?", re.I)),
    ("batch processing", re.compile(r"batch\s*(?:process|size|ing)", re.I)),
    ("rate limiting", re.compile(r"rate\s*limit(?:ing)?|throttl(?:e|ing)", re.I)),
    ("circuit breaker", re.compile(r"circuit\s*breaker", re.I)),
    ("connection pooling", re.compile(r"connection\s*pool(?:ing)?", re.I)),
    ("pagination", re.compile(r"pagination|paginate", re.I)),
    ("indexing", re.compile(r"(?:create|add|use)\s*index(?:es|ing)?", re.I)),
    ("sharding", re.compile(r"shard(?:ing)?", re.I)),
    ("data augmentation", re.compile(r"data\s*augmentation|augment\s*data", re.I)),
    ("learning rate scheduling", re.compile(r"learning\s*rate\s*schedul", re.I)),
    ("gradient accumulation", re.compile(r"gradient\s*accumulation", re.I)),
]


class HeuristicExtractor:
    """Extract entities and relations from training pairs using rules and dictionaries."""

    def extract_from_pair(self, pair: dict, source_doc: str = "") -> list[Triple]:
        """Extract triples from a single training pair."""
        triples: list[Triple] = []

        # Combine text fields for analysis
        instruction = pair.get("instruction", "")
        output = pair.get("output", "")
        category = pair.get("category", "")
        text = f"{instruction} {output}"

        # 1. Category as entity
        if category:
            cat_triples = self._extract_category(category, text, source_doc)
            triples.extend(cat_triples)

        # 2. Technologies
        found_techs = self._extract_technologies(text, source_doc)
        triples.extend(found_techs)

        # 3. Patterns
        found_patterns = self._extract_patterns(text, source_doc)
        triples.extend(found_patterns)

        # 4. Error types
        found_errors = self._extract_errors(text, source_doc)
        triples.extend(found_errors)

        # 5. Strategies
        found_strategies = self._extract_strategies(text, source_doc)
        triples.extend(found_strategies)

        # 6. Co-occurrence relations between technologies
        tech_names = [t.subject_name for t in found_techs if t.subject_type == EntityType.TECHNOLOGY]
        triples.extend(self._create_cooccurrence_relations(
            tech_names, EntityType.TECHNOLOGY, source_doc,
        ))

        return triples

    def _extract_category(self, category: str, text: str, source_doc: str) -> list[Triple]:
        """Extract category entity and link technologies/patterns to it."""
        triples = []
        cat_name = category.strip().lower()
        if not cat_name:
            return triples

        # Find technologies mentioned and link to category
        for tech_name in self._find_technologies(text):
            triples.append(Triple(
                subject_name=tech_name,
                subject_type=EntityType.TECHNOLOGY,
                relation_type=RelationType.RELATED_TO,
                object_name=cat_name,
                object_type=EntityType.CATEGORY,
                source_doc=source_doc,
                confidence=0.6,
            ))

        return triples

    def _extract_technologies(self, text: str, source_doc: str) -> list[Triple]:
        """Extract technology entities."""
        triples = []
        found = self._find_technologies(text)

        for tech_name in found:
            triples.append(Triple(
                subject_name=tech_name,
                subject_type=EntityType.TECHNOLOGY,
                relation_type=RelationType.RELATED_TO,
                object_name=tech_name,
                object_type=EntityType.TECHNOLOGY,
                source_doc=source_doc,
                confidence=0.8,
            ))

        return triples

    def _extract_patterns(self, text: str, source_doc: str) -> list[Triple]:
        """Extract design pattern entities."""
        triples = []
        text_lower = text.lower()

        for pattern_name, aliases in KNOWN_PATTERNS.items():
            all_names = [pattern_name] + aliases
            for name in all_names:
                if name.lower() in text_lower:
                    triples.append(Triple(
                        subject_name=pattern_name,
                        subject_type=EntityType.PATTERN,
                        relation_type=RelationType.RELATED_TO,
                        object_name=pattern_name,
                        object_type=EntityType.PATTERN,
                        source_doc=source_doc,
                        confidence=0.7,
                    ))
                    break

        return triples

    def _extract_errors(self, text: str, source_doc: str) -> list[Triple]:
        """Extract error type entities and link fixes."""
        triples = []
        found_errors: list[str] = []

        for error_name, pattern in KNOWN_ERROR_PATTERNS:
            if pattern.search(text):
                found_errors.append(error_name)
                triples.append(Triple(
                    subject_name=error_name,
                    subject_type=EntityType.ERROR_TYPE,
                    relation_type=RelationType.RELATED_TO,
                    object_name=error_name,
                    object_type=EntityType.ERROR_TYPE,
                    source_doc=source_doc,
                    confidence=0.7,
                ))

        # Link strategies that fix errors
        found_strategies = self._find_strategies(text)
        for error_name in found_errors:
            for strategy_name in found_strategies:
                triples.append(Triple(
                    subject_name=strategy_name,
                    subject_type=EntityType.STRATEGY,
                    relation_type=RelationType.FIXES,
                    object_name=error_name,
                    object_type=EntityType.ERROR_TYPE,
                    source_doc=source_doc,
                    confidence=0.6,
                ))

        return triples

    def _extract_strategies(self, text: str, source_doc: str) -> list[Triple]:
        """Extract strategy entities."""
        triples = []
        for strategy_name in self._find_strategies(text):
            triples.append(Triple(
                subject_name=strategy_name,
                subject_type=EntityType.STRATEGY,
                relation_type=RelationType.RELATED_TO,
                object_name=strategy_name,
                object_type=EntityType.STRATEGY,
                source_doc=source_doc,
                confidence=0.7,
            ))
        return triples

    # --- Helpers ---

    def _find_technologies(self, text: str) -> list[str]:
        """Find all known technologies mentioned in text."""
        found = []
        text_lower = text.lower()

        for tech_name, aliases in KNOWN_TECHNOLOGIES.items():
            all_names = [tech_name] + aliases
            for name in all_names:
                # Word boundary check to avoid partial matches
                pattern = r'\b' + re.escape(name.lower()) + r'\b'
                if re.search(pattern, text_lower):
                    found.append(tech_name)
                    break

        return found

    def _find_strategies(self, text: str) -> list[str]:
        """Find all known strategies mentioned in text."""
        found = []
        for strategy_name, pattern in KNOWN_STRATEGIES:
            if pattern.search(text):
                found.append(strategy_name)
        return found

    def _create_cooccurrence_relations(
        self,
        names: list[str],
        entity_type: EntityType,
        source_doc: str,
    ) -> list[Triple]:
        """Create RELATED_TO relations between co-occurring entities."""
        triples = []
        unique = list(set(names))

        for i, name_a in enumerate(unique):
            for name_b in unique[i + 1:]:
                if name_a == name_b:
                    continue
                triples.append(Triple(
                    subject_name=name_a,
                    subject_type=entity_type,
                    relation_type=RelationType.RELATED_TO,
                    object_name=name_b,
                    object_type=entity_type,
                    source_doc=source_doc,
                    confidence=0.5,
                ))

        return triples

    def extract_from_decision(self, decision: dict, source_doc: str = "") -> list[Triple]:
        """Extract triples from a decision document."""
        triples = []
        topic = decision.get("topic", decision.get("title", ""))
        chosen = decision.get("chosen_option", decision.get("decision", ""))
        lesson = decision.get("lesson_learned", "")

        text = f"{topic} {chosen} {lesson}"

        if topic:
            # Topic as concept
            for tech in self._find_technologies(text):
                triples.append(Triple(
                    subject_name=tech,
                    subject_type=EntityType.TECHNOLOGY,
                    relation_type=RelationType.RELATED_TO,
                    object_name=topic.strip().lower(),
                    object_type=EntityType.CONCEPT,
                    source_doc=source_doc,
                    confidence=0.7,
                ))

        if chosen:
            for strategy in self._find_strategies(text):
                triples.append(Triple(
                    subject_name=strategy,
                    subject_type=EntityType.STRATEGY,
                    relation_type=RelationType.LEARNED_FROM,
                    object_name=topic.strip().lower() if topic else "decision",
                    object_type=EntityType.CONCEPT,
                    source_doc=source_doc,
                    confidence=0.7,
                ))

        return triples

    def extract_from_learning(self, learning: dict, source_doc: str = "") -> list[Triple]:
        """Extract triples from a learning document."""
        triples = []
        topic = learning.get("topic", learning.get("title", ""))
        applicable_to = learning.get("applicable_to", [])

        text = f"{topic} {' '.join(applicable_to) if isinstance(applicable_to, list) else applicable_to}"

        if topic:
            for tech in self._find_technologies(text):
                triples.append(Triple(
                    subject_name=topic.strip().lower(),
                    subject_type=EntityType.CONCEPT,
                    relation_type=RelationType.RELATED_TO,
                    object_name=tech,
                    object_type=EntityType.TECHNOLOGY,
                    source_doc=source_doc,
                    confidence=0.6,
                ))

        return triples

    def extract_from_auto_capture(self, record: dict, source_doc: str = "") -> list[Triple]:
        """Extract triples from an auto-capture record.

        Auto-captures lack reasoning, so confidence scores are lower (0.4-0.5).
        Enriched captures with reasoning get a confidence boost (0.6-0.7).
        We infer technologies from file extensions and description text.
        """
        triples: list[Triple] = []

        file_modified = record.get("file_modified", "")
        description = record.get("description", "")
        project = record.get("project", "")
        reasoning = record.get("reasoning", "")

        # Confidence boost based on enrichment
        enrichment_confidence = record.get("enrichment_confidence", "")
        if enrichment_confidence == "high":
            base_confidence = 0.7
            cooccurrence_confidence = 0.6
        elif enrichment_confidence == "medium":
            base_confidence = 0.6
            cooccurrence_confidence = 0.55
        else:
            base_confidence = 0.4
            cooccurrence_confidence = 0.5

        # Include reasoning in text for extraction when available
        text = f"{file_modified} {description}"
        if reasoning:
            text = f"{text} {reasoning}"

        # 1. Infer technology from file extension
        ext_techs: list[str] = []
        if file_modified:
            from pathlib import PurePosixPath
            ext = PurePosixPath(file_modified).suffix.lower()
            if ext in FILE_EXT_TO_TECH:
                tech = FILE_EXT_TO_TECH[ext]
                ext_techs.append(tech)
                triples.append(Triple(
                    subject_name=tech,
                    subject_type=EntityType.TECHNOLOGY,
                    relation_type=RelationType.RELATED_TO,
                    object_name=tech,
                    object_type=EntityType.TECHNOLOGY,
                    source_doc=source_doc,
                    confidence=base_confidence,
                ))

        # 2. Technologies from description (and reasoning if available)
        desc_techs = self._find_technologies(text)
        for tech in desc_techs:
            triples.append(Triple(
                subject_name=tech,
                subject_type=EntityType.TECHNOLOGY,
                relation_type=RelationType.RELATED_TO,
                object_name=tech,
                object_type=EntityType.TECHNOLOGY,
                source_doc=source_doc,
                confidence=base_confidence,
            ))

        # 3. Patterns from description (and reasoning)
        search_text = f"{description} {reasoning}" if reasoning else description
        for pattern_triple in self._extract_patterns(search_text, source_doc):
            triples.append(Triple(
                subject_name=pattern_triple.subject_name,
                subject_type=pattern_triple.subject_type,
                relation_type=pattern_triple.relation_type,
                object_name=pattern_triple.object_name,
                object_type=pattern_triple.object_type,
                source_doc=source_doc,
                confidence=base_confidence,
            ))

        # 4. Strategies from description (and reasoning)
        for strat_triple in self._extract_strategies(search_text, source_doc):
            triples.append(Triple(
                subject_name=strat_triple.subject_name,
                subject_type=strat_triple.subject_type,
                relation_type=strat_triple.relation_type,
                object_name=strat_triple.object_name,
                object_type=strat_triple.object_type,
                source_doc=source_doc,
                confidence=base_confidence,
            ))

        # 5. Project USES technology
        if project:
            all_techs = list(set(ext_techs + desc_techs))
            for tech in all_techs:
                triples.append(Triple(
                    subject_name=project.strip().lower(),
                    subject_type=EntityType.CONCEPT,
                    relation_type=RelationType.USES,
                    object_name=tech,
                    object_type=EntityType.TECHNOLOGY,
                    source_doc=source_doc,
                    confidence=cooccurrence_confidence,
                ))

        # 6. Co-occurrence between all technologies
        all_techs = list(set(ext_techs + desc_techs))
        triples.extend(self._create_cooccurrence_relations(
            all_techs, EntityType.TECHNOLOGY, source_doc,
        ))

        return triples
