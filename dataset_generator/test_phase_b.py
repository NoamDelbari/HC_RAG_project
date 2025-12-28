"""
Test script for Phase B: Query Template Development and LLM Prompt Engineering

Tests:
1. Config file loading and structure validation
2. PromptBuilder functionality
3. OutputParser functionality
4. QueryGenerator (if API keys available)
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env.example in the same directory as this script
env_path = Path(__file__).parent / ".env.example"
load_dotenv(env_path)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_config_loading():
    """Test that all config files load correctly."""
    print("\n" + "=" * 60)
    print("TEST 1: Config File Loading")
    print("=" * 60)

    config_dir = Path(__file__).parent / "config"
    config_files = [
        "cluster_types.json",
        "query_templates.json",
        "difficulty_calibration.json",
        "llm_prompts.json"
    ]

    all_passed = True
    for filename in config_files:
        filepath = config_dir / filename
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"  [PASS] {filename} - loaded successfully ({len(str(data))} chars)")
        except FileNotFoundError:
            print(f"  [FAIL] {filename} - file not found")
            all_passed = False
        except json.JSONDecodeError as e:
            print(f"  [FAIL] {filename} - JSON parse error: {e}")
            all_passed = False

    return all_passed


def test_config_structure():
    """Test that config files have expected structure."""
    print("\n" + "=" * 60)
    print("TEST 2: Config Structure Validation")
    print("=" * 60)

    config_dir = Path(__file__).parent / "config"
    all_passed = True

    # Test cluster_types.json
    with open(config_dir / "cluster_types.json", "r", encoding="utf-8") as f:
        cluster_types = json.load(f)

    expected_types = ["categorical", "creative_ensemble", "award_recognition",
                      "organizational", "event_temporal", "geographic_spatial", "relational"]

    for ct in expected_types:
        if ct in cluster_types.get("cluster_types", {}):
            print(f"  [PASS] cluster_types.json has '{ct}'")
        else:
            print(f"  [FAIL] cluster_types.json missing '{ct}'")
            all_passed = False

    # Test query_templates.json
    with open(config_dir / "query_templates.json", "r", encoding="utf-8") as f:
        templates = json.load(f)

    template_count = 0
    for cluster_type, difficulties in templates.get("templates", {}).items():
        for difficulty, template_list in difficulties.items():
            template_count += len(template_list)

    if template_count >= 21:
        print(f"  [PASS] query_templates.json has {template_count} templates")
    else:
        print(f"  [FAIL] query_templates.json only has {template_count} templates (expected 21+)")
        all_passed = False

    # Test difficulty_calibration.json
    with open(config_dir / "difficulty_calibration.json", "r", encoding="utf-8") as f:
        difficulty = json.load(f)

    expected_dims = ["lexical_overlap", "reasoning_hops", "indirection_level"]
    for dim in expected_dims:
        if dim in difficulty.get("difficulty_dimensions", {}):
            print(f"  [PASS] difficulty_calibration.json has dimension '{dim}'")
        else:
            print(f"  [FAIL] difficulty_calibration.json missing dimension '{dim}'")
            all_passed = False

    # Test llm_prompts.json
    with open(config_dir / "llm_prompts.json", "r", encoding="utf-8") as f:
        prompts = json.load(f)

    expected_sections = ["system_context", "cluster_type_prompts", "difficulty_modifiers", "output_format"]
    for section in expected_sections:
        if section in prompts:
            print(f"  [PASS] llm_prompts.json has section '{section}'")
        else:
            print(f"  [FAIL] llm_prompts.json missing section '{section}'")
            all_passed = False

    return all_passed


def test_prompt_builder():
    """Test PromptBuilder functionality."""
    print("\n" + "=" * 60)
    print("TEST 3: PromptBuilder Functionality")
    print("=" * 60)

    from query_generator import PromptBuilder

    builder = PromptBuilder(config_dir="config")
    all_passed = True

    # Test system context
    sys_ctx = builder.build_system_context()
    if "benchmark query generator" in sys_ctx.lower():
        print("  [PASS] build_system_context() returns valid content")
    else:
        print("  [FAIL] build_system_context() missing expected content")
        all_passed = False

    # Test cluster context
    test_cluster = {
        "cluster_id": "test_001",
        "cluster_type": "creative_ensemble",
        "binding_description": "Cast of Inception",
        "entities": [
            {"qid": "Q134773", "label": "Leonardo DiCaprio", "description": "American actor"},
            {"qid": "Q188018", "label": "Tom Hardy", "description": "English actor"}
        ]
    }

    cluster_ctx = builder.build_cluster_context(test_cluster)
    if "Leonardo DiCaprio" in cluster_ctx and "Q134773" in cluster_ctx:
        print("  [PASS] build_cluster_context() includes entity data")
    else:
        print("  [FAIL] build_cluster_context() missing entity data")
        all_passed = False

    # Test cluster type prompt
    for cluster_type in ["categorical", "creative_ensemble", "award_recognition"]:
        type_prompt = builder.build_cluster_type_prompt(cluster_type)
        if len(type_prompt) > 50:
            print(f"  [PASS] build_cluster_type_prompt('{cluster_type}') returns content")
        else:
            print(f"  [FAIL] build_cluster_type_prompt('{cluster_type}') too short")
            all_passed = False

    # Test difficulty modifier
    for difficulty in ["easy", "medium", "hard"]:
        diff_mod = builder.build_difficulty_modifier(difficulty)
        if difficulty.upper() in diff_mod:
            print(f"  [PASS] build_difficulty_modifier('{difficulty}') returns content")
        else:
            print(f"  [FAIL] build_difficulty_modifier('{difficulty}') missing difficulty label")
            all_passed = False

    # Test complete prompt
    complete_prompt = builder.build_complete_prompt(
        cluster_data=test_cluster,
        difficulty="medium",
        target_k=5,
        previous_queries=["Who are the actors in Inception?"]
    )

    checks = [
        ("system context", "benchmark query generator" in complete_prompt.lower()),
        ("cluster data", "Leonardo DiCaprio" in complete_prompt),
        ("difficulty modifier", "MEDIUM" in complete_prompt),
        ("previous queries", "actors in Inception" in complete_prompt),
        ("output format", "JSON" in complete_prompt)
    ]

    for check_name, passed in checks:
        if passed:
            print(f"  [PASS] Complete prompt includes {check_name}")
        else:
            print(f"  [FAIL] Complete prompt missing {check_name}")
            all_passed = False

    print(f"\n  Complete prompt length: {len(complete_prompt)} characters")

    return all_passed


def test_output_parser():
    """Test OutputParser functionality."""
    print("\n" + "=" * 60)
    print("TEST 4: OutputParser Functionality")
    print("=" * 60)

    from query_generator import OutputParser

    parser = OutputParser()
    all_passed = True

    # Test valid JSON parsing
    valid_output = """```json
{
  "query": "What nationalities do the lead actors in Nolan's 2010 thriller have?",
  "expected_answer_type": "list",
  "entity_coverage": ["Q134773", "Q188018"],
  "difficulty_justification": {
    "lexical_overlap_strategy": "Used descriptive reference",
    "reasoning_hops": 2,
    "indirection_explanation": "Film referenced descriptively"
  },
  "answerable_from_passages": true
}
```"""

    result = parser.parse(valid_output)
    if result.is_valid and result.query.endswith("?"):
        print("  [PASS] Parser handles valid JSON in code block")
    else:
        print(f"  [FAIL] Parser failed on valid JSON: {result.validation_errors}")
        all_passed = False

    # Test raw JSON parsing
    raw_json = """{
  "query": "Who received the Nobel Prize in Physics in 2020?",
  "expected_answer_type": "list",
  "entity_coverage": ["Q937", "Q123"],
  "difficulty_justification": {
    "lexical_overlap_strategy": "Direct entity names",
    "reasoning_hops": 1,
    "indirection_explanation": "Direct reference"
  },
  "answerable_from_passages": true
}"""

    result = parser.parse(raw_json)
    if result.is_valid:
        print("  [PASS] Parser handles raw JSON")
    else:
        print(f"  [FAIL] Parser failed on raw JSON: {result.validation_errors}")
        all_passed = False

    # Test invalid query (no question mark)
    no_question = """{
  "query": "List the actors in Inception",
  "expected_answer_type": "list",
  "entity_coverage": ["Q134773"],
  "difficulty_justification": {},
  "answerable_from_passages": true
}"""

    result = parser.parse(no_question)
    if not result.is_valid and "question mark" in str(result.validation_errors).lower():
        print("  [PASS] Parser detects missing question mark")
    else:
        print(f"  [FAIL] Parser should reject query without question mark")
        all_passed = False

    # Test malformed JSON
    bad_json = "This is not JSON at all"
    result = parser.parse(bad_json)
    if not result.is_valid:
        print("  [PASS] Parser handles malformed JSON gracefully")
    else:
        print("  [FAIL] Parser should fail on malformed JSON")
        all_passed = False

    # Test partial JSON in text
    mixed_output = """Here's my query:

{
  "query": "Which universities did the Nobel laureates attend?",
  "expected_answer_type": "list",
  "entity_coverage": ["Q937"],
  "difficulty_justification": {
    "lexical_overlap_strategy": "Paraphrased",
    "reasoning_hops": 2,
    "indirection_explanation": "Descriptive"
  },
  "answerable_from_passages": true
}

That should work!"""

    result = parser.parse(mixed_output)
    if result.is_valid and "universities" in result.query.lower():
        print("  [PASS] Parser extracts JSON from mixed text")
    else:
        print(f"  [FAIL] Parser should extract JSON from mixed text")
        all_passed = False

    return all_passed


def test_query_generator_init():
    """Test QueryGenerator initialization (without API calls)."""
    print("\n" + "=" * 60)
    print("TEST 5: QueryGenerator Initialization")
    print("=" * 60)

    from query_generator import QueryGenerator, GenerationConfig
    import os

    all_passed = True

    # Test with default config
    config = GenerationConfig()
    print(f"  Default provider: {config.provider}")
    print(f"  Default model: {config.model}")
    print(f"  Default temperature: {config.temperature}")

    # Test generator creation
    try:
        generator = QueryGenerator(config=config, config_dir="config")
        print("  [PASS] QueryGenerator created successfully")

        # Check if client is available
        if generator.client is not None:
            print(f"  [PASS] LLM client initialized ({config.provider})")
        else:
            api_key_name = "ANTHROPIC_API_KEY" if config.provider == "anthropic" else "OPENAI_API_KEY"
            if os.getenv(api_key_name):
                print(f"  [WARN] API key set but client failed to initialize")
            else:
                print(f"  [INFO] No {api_key_name} set - client not initialized (expected)")

    except Exception as e:
        print(f"  [FAIL] QueryGenerator creation failed: {e}")
        all_passed = False

    return all_passed


def test_live_generation():
    """Test actual query generation with API (if available)."""
    print("\n" + "=" * 60)
    print("TEST 6: Live Query Generation (API Test)")
    print("=" * 60)

    import os
    from query_generator import QueryGenerator, GenerationConfig

    # Check for API keys
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not anthropic_key and not openai_key:
        print("  [SKIP] No API keys available - skipping live test")
        print("         Set ANTHROPIC_API_KEY or OPENAI_API_KEY to run this test")
        return True

    # Choose provider based on available key - get model from environment variable
    import os
    if anthropic_key:
        model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        config = GenerationConfig(provider="anthropic", model=model)
        print(f"  Using Anthropic API ({model})")
    else:
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        config = GenerationConfig(provider="openai", model=model)
        print(f"  Using OpenAI API ({model})")

    generator = QueryGenerator(config=config, config_dir="config")

    if generator.client is None:
        print("  [FAIL] Client not initialized despite API key being set")
        return False

    # Test cluster
    test_cluster = {
        "cluster_id": "test_creative_001",
        "cluster_type": "creative_ensemble",
        "binding_description": "Main cast members of the film Inception (2010)",
        "entities": [
            {
                "qid": "Q134773",
                "label": "Leonardo DiCaprio",
                "description": "American actor and film producer",
                "key_facts": "Born 1974, Oscar winner for The Revenant"
            },
            {
                "qid": "Q188018",
                "label": "Tom Hardy",
                "description": "English actor",
                "key_facts": "Born 1977, known for Bane, Mad Max"
            },
            {
                "qid": "Q232163",
                "label": "Joseph Gordon-Levitt",
                "description": "American actor and filmmaker",
                "key_facts": "Born 1981, former child actor"
            },
            {
                "qid": "Q174843",
                "label": "Marion Cotillard",
                "description": "French actress",
                "key_facts": "Born 1975, Oscar winner for La Vie en Rose"
            }
        ]
    }

    all_passed = True

    # Test each difficulty level
    for difficulty in ["easy", "medium", "hard"]:
        print(f"\n  Generating {difficulty.upper()} query...")
        try:
            result = generator.generate_query(
                cluster_data=test_cluster,
                difficulty=difficulty,
                target_k=4
            )

            if result.is_valid:
                print(f"  [PASS] {difficulty.upper()}: {result.query[:80]}...")
                print(f"         Answer type: {result.expected_answer_type}")
                print(f"         Entities covered: {len(result.entity_coverage)}")
            else:
                print(f"  [FAIL] {difficulty.upper()}: {result.validation_errors}")
                all_passed = False

        except Exception as e:
            print(f"  [FAIL] {difficulty.upper()}: Exception - {e}")
            all_passed = False

    return all_passed


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PHASE B TEST SUITE")
    print("Cross-Entity Query Generation")
    print("=" * 60)

    results = {}

    # Run tests
    results["Config Loading"] = test_config_loading()
    results["Config Structure"] = test_config_structure()
    results["PromptBuilder"] = test_prompt_builder()
    results["OutputParser"] = test_output_parser()
    results["Generator Init"] = test_query_generator_init()
    results["Live Generation"] = test_live_generation()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  All tests passed! Phase B is ready.")
        return 0
    else:
        print("\n  Some tests failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
