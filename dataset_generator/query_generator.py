"""
Cross-Entity Query Generator

This module generates queries for cross-entity clusters using LLM prompts.
It implements the prompt architecture from llm_prompts.json and handles
output parsing and validation.

Phase B.4: LLM Prompt Engineering Implementation
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env.example in the same directory as this script
env_path = Path(__file__).parent / ".env.example"
load_dotenv(env_path)


@dataclass
class QueryOutput:
    """Structured output from query generation."""
    query: str
    expected_answer_type: str
    entity_coverage: list
    difficulty_justification: dict
    answerable_from_passages: bool
    notes: str = ""

    # Validation metadata
    is_valid: bool = True
    validation_errors: list = field(default_factory=list)
    validation_warnings: list = field(default_factory=list)


@dataclass
class GenerationConfig:
    """Configuration for query generation."""
    provider: str = "anthropic"  # or "openai"
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.7
    max_tokens: int = 500
    max_retries: int = 3


class PromptBuilder:
    """Builds prompts from templates and context."""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.prompts_config = self._load_config("llm_prompts.json")
        self.templates_config = self._load_config("query_templates.json")
        self.difficulty_config = self._load_config("difficulty_calibration.json")
        self.cluster_types_config = self._load_config("cluster_types.json")

    def _load_config(self, filename: str) -> dict:
        """Load a JSON config file."""
        filepath = self.config_dir / filename
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def build_system_context(self) -> str:
        """Build the system context block."""
        ctx = self.prompts_config.get("system_context", {})

        lines = [
            f"Role: {ctx.get('role', '')}",
            f"\nPurpose: {ctx.get('purpose', '')}",
            "\nQuality Expectations:"
        ]
        for exp in ctx.get("quality_expectations", []):
            lines.append(f"- {exp}")

        lines.append("\nConstraints:")
        for con in ctx.get("constraints", []):
            lines.append(f"- {con}")

        return "\n".join(lines)

    def build_cluster_context(self, cluster_data: dict) -> str:
        """Build the cluster context block from cluster data."""
        lines = [
            f"Cluster ID: {cluster_data.get('cluster_id', 'unknown')}",
            f"Cluster Type: {cluster_data.get('cluster_type', 'unknown')}",
            f"Binding Property: {cluster_data.get('binding_description', '')}",
            f"Entity Count: {len(cluster_data.get('entities', []))}"
        ]

        # Add temporal constraint if present
        temporal_constraint = cluster_data.get("temporal_constraint", "")
        if temporal_constraint and temporal_constraint != "subset of entities":
            lines.append(f"\n*** TEMPORAL CONSTRAINT: {temporal_constraint} ***")
            lines.append("IMPORTANT: Your query MUST include this temporal constraint!")
            lines.append("The query should ONLY be answerable by entities matching this constraint.")

        lines.append("\nEntities:")

        for entity in cluster_data.get("entities", []):
            lines.append(f"\n  - QID: {entity.get('qid', '')}")
            lines.append(f"    Label: {entity.get('label', '')}")
            lines.append(f"    Description: {entity.get('description', '')}")
            # Include year info from properties
            props = entity.get("properties", {})
            year_info = props.get("award_year") or props.get("release_year") or props.get("start_year")
            if year_info:
                lines.append(f"    Year: {year_info}")
            if entity.get("key_facts"):
                lines.append(f"    Key Facts: {entity.get('key_facts', '')}")

        return "\n".join(lines)

    def build_cluster_type_prompt(self, cluster_type: str) -> str:
        """Build cluster-type-specific guidance."""
        type_prompts = self.prompts_config.get("cluster_type_prompts", {})
        prompt_data = type_prompts.get(cluster_type, {})

        if not prompt_data:
            return f"Cluster type: {cluster_type}"

        lines = [
            prompt_data.get("context_emphasis", ""),
            f"\nBinding: {prompt_data.get('binding_description', '')}",
            "\nQuery Focus Suggestions:"
        ]
        for suggestion in prompt_data.get("query_focus_suggestions", []):
            lines.append(f"- {suggestion}")

        lines.append("\nGood Query Angles:")
        for angle in prompt_data.get("good_angles", []):
            lines.append(f"- {angle}")

        lines.append("\nAvoid:")
        for avoid in prompt_data.get("avoid", []):
            lines.append(f"- {avoid}")

        return "\n".join(lines)

    def build_difficulty_modifier(self, difficulty: str) -> str:
        """Build difficulty-specific instructions."""
        modifiers = self.prompts_config.get("difficulty_modifiers", {})
        mod_data = modifiers.get(difficulty, {})

        if not mod_data:
            return f"Difficulty: {difficulty}"

        lines = [
            mod_data.get("instruction", ""),
            "\nRequirements:"
        ]
        for req in mod_data.get("requirements", []):
            lines.append(f"- {req}")

        lines.append("\nPhrasing Guidance:")
        for guidance in mod_data.get("phrasing_guidance", []):
            lines.append(f"- {guidance}")

        lines.append("\nExample Patterns:")
        for pattern in mod_data.get("example_patterns", []):
            lines.append(f"- {pattern}")

        lines.append("\nAvoid:")
        for avoid in mod_data.get("avoid", []):
            lines.append(f"- {avoid}")

        return "\n".join(lines)

    def build_diversity_instructions(self, previous_queries: list = None) -> str:
        """Build diversity instructions with optional previous queries."""
        diversity = self.prompts_config.get("diversity_instructions", {})

        lines = [diversity.get("base", "")]

        dims = diversity.get("dimensions", {})
        for dim_name, dim_data in dims.items():
            lines.append(f"\n{dim_name.replace('_', ' ').title()}:")
            lines.append(f"  {dim_data.get('instruction', '')}")
            if "examples" in dim_data:
                for ex in dim_data["examples"]:
                    lines.append(f"  - {ex}")
            if "rotation" in dim_data:
                for r in dim_data["rotation"]:
                    lines.append(f"  - {r}")

        if previous_queries:
            lines.append("\n\nPreviously Generated Queries (DO NOT repeat similar patterns):")
            for i, pq in enumerate(previous_queries[-5:], 1):  # Show last 5
                lines.append(f"  {i}. {pq}")

        return "\n".join(lines)

    def build_output_format(self) -> str:
        """Build output format specification."""
        return """Provide your response as a JSON object with these fields:
{
  "query": "The natural language query (must end with ?)",
  "expected_answer_type": "list | count | comparison | aggregation | narrative",
  "entity_coverage": ["QID1", "QID2", ...],
  "difficulty_justification": {
    "lexical_overlap_strategy": "How you achieved target overlap",
    "reasoning_hops": 1-4,
    "indirection_explanation": "How entities are referenced"
  },
  "answerable_from_passages": true/false,
  "notes": "Optional notes"
}"""

    def build_template_spec(self, cluster_type: str, difficulty: str) -> str:
        """Get the template specification for cluster type and difficulty."""
        templates = self.templates_config.get("templates", {})
        type_templates = templates.get(cluster_type, {})
        difficulty_templates = type_templates.get(difficulty, [])

        if not difficulty_templates:
            return "No specific template available. Generate a natural query."

        template = difficulty_templates[0]  # Use first template

        lines = [
            f"Template ID: {template.get('template_id', '')}",
            f"Pattern: {template.get('pattern', '')}",
            f"K Range: {template.get('k_range', {})}",
            "\nSlots:"
        ]
        for slot_name, slot_desc in template.get("slots", {}).items():
            lines.append(f"  - {slot_name}: {slot_desc}")

        lines.append(f"\nAnswer Type: {template.get('answer_spec', {}).get('type', 'list')}")

        return "\n".join(lines)

    def build_complete_prompt(
        self,
        cluster_data: dict,
        difficulty: str,
        target_k: int,
        previous_queries: list = None
    ) -> str:
        """Build the complete prompt from all blocks."""
        cluster_type = cluster_data.get("cluster_type", "categorical")
        temporal_constraint = cluster_data.get("temporal_constraint", "")

        sections = [
            "# System Context",
            self.build_system_context(),
            "\n---\n",
            "# Cluster Information",
            self.build_cluster_context(cluster_data),
            "\n---\n",
            "# Cluster Type Guidance",
            self.build_cluster_type_prompt(cluster_type),
            "\n---\n",
            "# Query Requirements",
            f"Target K (passages needed): {target_k}",
        ]

        # Add constraint-specific instructions
        if temporal_constraint and temporal_constraint != "subset of entities":
            sections.extend([
                "",
                "*** CRITICAL CONSTRAINT REQUIREMENT ***",
                f"The query MUST include the temporal constraint: {temporal_constraint}",
                "This ensures the query is answerable by EXACTLY the provided entities.",
                "",
                "Examples of constraint-based queries:",
                f"- 'Who won the Nobel Prize in Physics {temporal_constraint}?'",
                f"- 'What films were released {temporal_constraint}?'",
                f"- 'Which people held this position {temporal_constraint}?'",
                "",
                "DO NOT generate generic queries like 'What films did X direct?' - these cover ALL entities.",
                "DO generate specific queries like 'What films did X direct between 2000 and 2005?' - these cover ONLY the constrained entities.",
                ""
            ])

        sections.extend([
            self.build_template_spec(cluster_type, difficulty),
            "\n",
            self.build_difficulty_modifier(difficulty),
            "\n---\n",
            "# Diversity Requirements",
            self.build_diversity_instructions(previous_queries),
            "\n---\n",
            "# Output Format",
            self.build_output_format(),
            "\n---\n",
            "Generate a high-quality query following all specifications above."
        ])

        return "\n".join(sections)


class OutputParser:
    """Parses and validates LLM query generation output."""

    def __init__(self):
        self.validation_rules = {
            "query": [
                ("ends_with_question_mark", self._check_question_mark),
                ("length_in_range", self._check_length),
            ],
            "entity_coverage": [
                ("is_list", self._check_is_list),
            ],
            "difficulty_justification": [
                ("has_required_fields", self._check_justification_fields),
            ]
        }

    def parse(self, raw_output: str) -> QueryOutput:
        """Parse raw LLM output into structured QueryOutput."""
        # Try to extract JSON from the response
        json_data = self._extract_json(raw_output)

        if json_data is None:
            return QueryOutput(
                query="",
                expected_answer_type="unknown",
                entity_coverage=[],
                difficulty_justification={},
                answerable_from_passages=False,
                is_valid=False,
                validation_errors=["Failed to parse JSON from output"]
            )

        # Create QueryOutput from parsed data
        output = QueryOutput(
            query=json_data.get("query", ""),
            expected_answer_type=json_data.get("expected_answer_type", "list"),
            entity_coverage=json_data.get("entity_coverage", []),
            difficulty_justification=json_data.get("difficulty_justification", {}),
            answerable_from_passages=json_data.get("answerable_from_passages", True),
            notes=json_data.get("notes", "")
        )

        # Validate
        self._validate(output)

        return output

    def _extract_json(self, text: str) -> Optional[dict]:
        """Extract JSON object from text."""
        # Try direct parse first
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try to find JSON block
        json_patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[\s\S]*\}'
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue

        return None

    def _validate(self, output: QueryOutput):
        """Run validation rules on output."""
        errors = []
        warnings = []

        # Query validation
        if not output.query:
            errors.append("Query is empty")
        else:
            if not output.query.strip().endswith("?"):
                errors.append("Query does not end with question mark")

            word_count = len(output.query.split())
            if word_count < 10:
                warnings.append(f"Query is short ({word_count} words)")
            elif word_count > 60:
                warnings.append(f"Query is long ({word_count} words)")

        # Entity coverage validation
        if not output.entity_coverage:
            warnings.append("No entity coverage specified")
        elif not isinstance(output.entity_coverage, list):
            errors.append("Entity coverage must be a list")

        # Difficulty justification validation
        if output.difficulty_justification:
            required_fields = ["lexical_overlap_strategy", "reasoning_hops", "indirection_explanation"]
            missing = [f for f in required_fields if f not in output.difficulty_justification]
            if missing:
                warnings.append(f"Missing difficulty justification fields: {missing}")
        else:
            warnings.append("No difficulty justification provided")

        output.validation_errors = errors
        output.validation_warnings = warnings
        output.is_valid = len(errors) == 0

    def _check_question_mark(self, value: str) -> bool:
        return value.strip().endswith("?")

    def _check_length(self, value: str) -> bool:
        word_count = len(value.split())
        return 10 <= word_count <= 60

    def _check_is_list(self, value) -> bool:
        return isinstance(value, list)

    def _check_justification_fields(self, value: dict) -> bool:
        required = ["lexical_overlap_strategy", "reasoning_hops", "indirection_explanation"]
        return all(f in value for f in required)


class QueryGenerator:
    """Main class for generating queries using LLM."""

    def __init__(self, config: GenerationConfig = None, config_dir: str = "config"):
        self.config = config or GenerationConfig()
        self.prompt_builder = PromptBuilder(config_dir)
        self.parser = OutputParser()
        self.client = None
        self._setup_client()

    def _setup_client(self):
        """Set up the LLM client based on provider."""
        if self.config.provider == "anthropic":
            try:
                import anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    self.client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                print("Warning: anthropic package not installed")
        elif self.config.provider == "openai":
            try:
                import openai
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.client = openai.OpenAI(api_key=api_key)
            except ImportError:
                print("Warning: openai package not installed")

    def generate_query(
        self,
        cluster_data: dict,
        difficulty: str,
        target_k: int,
        previous_queries: list = None
    ) -> QueryOutput:
        """Generate a single query for a cluster."""
        prompt = self.prompt_builder.build_complete_prompt(
            cluster_data=cluster_data,
            difficulty=difficulty,
            target_k=target_k,
            previous_queries=previous_queries
        )

        # Try generation with retries
        for attempt in range(self.config.max_retries):
            try:
                raw_output = self._call_llm(prompt)
                output = self.parser.parse(raw_output)

                if output.is_valid:
                    return output

                # If invalid, modify prompt and retry
                if attempt < self.config.max_retries - 1:
                    prompt += "\n\nIMPORTANT: Ensure your response is valid JSON and the query ends with a question mark."
                    time.sleep(0.5)

            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    return QueryOutput(
                        query="",
                        expected_answer_type="unknown",
                        entity_coverage=[],
                        difficulty_justification={},
                        answerable_from_passages=False,
                        is_valid=False,
                        validation_errors=[f"Generation failed: {str(e)}"]
                    )
                time.sleep(1)

        return output

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API."""
        if self.client is None:
            raise RuntimeError("LLM client not configured. Set API key environment variable.")

        if self.config.provider == "anthropic":
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        elif self.config.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content

        raise ValueError(f"Unknown provider: {self.config.provider}")

    def generate_batch(
        self,
        cluster_data: dict,
        difficulties: list = None,
        target_k: int = None,
        queries_per_difficulty: int = 1
    ) -> list:
        """Generate multiple queries for a cluster across difficulties."""
        difficulties = difficulties or ["easy", "medium", "hard"]
        target_k = target_k or len(cluster_data.get("entities", []))

        results = []
        previous_queries = []

        for difficulty in difficulties:
            for i in range(queries_per_difficulty):
                # Increase temperature for later queries
                if i > 0:
                    self.config.temperature = min(0.9, self.config.temperature + 0.1)

                output = self.generate_query(
                    cluster_data=cluster_data,
                    difficulty=difficulty,
                    target_k=target_k,
                    previous_queries=previous_queries
                )

                if output.is_valid:
                    previous_queries.append(output.query)

                results.append({
                    "difficulty": difficulty,
                    "output": output,
                    "cluster_id": cluster_data.get("cluster_id")
                })

                time.sleep(0.5)  # Rate limiting

        return results


def query_output_to_dict(output: QueryOutput) -> dict:
    """Convert QueryOutput to dictionary."""
    return {
        "query": output.query,
        "expected_answer_type": output.expected_answer_type,
        "entity_coverage": output.entity_coverage,
        "difficulty_justification": output.difficulty_justification,
        "answerable_from_passages": output.answerable_from_passages,
        "notes": output.notes,
        "is_valid": output.is_valid,
        "validation_errors": output.validation_errors,
        "validation_warnings": output.validation_warnings
    }


if __name__ == "__main__":
    # Test the prompt builder
    builder = PromptBuilder()

    # Sample cluster data
    test_cluster = {
        "cluster_id": "test_creative_001",
        "cluster_type": "creative_ensemble",
        "binding_description": "Cast members of Inception (2010)",
        "entities": [
            {
                "qid": "Q134773",
                "label": "Leonardo DiCaprio",
                "description": "American actor",
                "key_facts": "Born 1974, Oscar winner"
            },
            {
                "qid": "Q188018",
                "label": "Tom Hardy",
                "description": "English actor",
                "key_facts": "Born 1977, known for Bane in Dark Knight Rises"
            },
            {
                "qid": "Q232163",
                "label": "Joseph Gordon-Levitt",
                "description": "American actor",
                "key_facts": "Born 1981, former child actor"
            }
        ]
    }

    # Build and display prompt
    print("=" * 60)
    print("SAMPLE PROMPT (Medium Difficulty)")
    print("=" * 60)
    prompt = builder.build_complete_prompt(
        cluster_data=test_cluster,
        difficulty="medium",
        target_k=3,
        previous_queries=None
    )
    print(prompt[:2000])  # First 2000 chars
    print("\n... (truncated)")

    # Test parser
    print("\n" + "=" * 60)
    print("PARSER TEST")
    print("=" * 60)

    test_output = """```json
{
  "query": "What nationalities do the lead actors in Nolan's 2010 psychological thriller have?",
  "expected_answer_type": "list",
  "entity_coverage": ["Q134773", "Q188018", "Q232163"],
  "difficulty_justification": {
    "lexical_overlap_strategy": "Used descriptive reference 'Nolan's 2010 psychological thriller' instead of 'Inception'",
    "reasoning_hops": 2,
    "indirection_explanation": "Film referenced descriptively, actors referenced by role"
  },
  "answerable_from_passages": true
}
```"""

    parser = OutputParser()
    result = parser.parse(test_output)

    print(f"Query: {result.query}")
    print(f"Valid: {result.is_valid}")
    print(f"Errors: {result.validation_errors}")
    print(f"Warnings: {result.validation_warnings}")
