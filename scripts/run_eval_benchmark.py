#!/usr/bin/env python3
"""Run evaluation benchmark against Ollama models.

Usage:
    python run_eval_benchmark.py --model qwen2.5-coder:7b
    python run_eval_benchmark.py --model qwen2.5-coder:7b --category code-review
    python run_eval_benchmark.py --compare qwen2.5-coder:7b,llama3:8b,qwen2.5-coder:7b
"""

import argparse
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

BENCHMARK_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "eval-benchmark"
CASES_PATH = BENCHMARK_PATH / "cases"
RESULTS_PATH = BENCHMARK_PATH / "results"


def query_ollama(model: str, prompt: str, timeout: int = 60) -> tuple[str, float]:
    """Query Ollama model and return response with latency."""
    start = time.time()
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        latency = time.time() - start
        return result.stdout.strip(), latency
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]", timeout
    except Exception as e:
        return f"[ERROR: {e}]", time.time() - start


def evaluate_response(response: str, case: dict) -> dict:
    """Evaluate a model response against expected behavior."""
    expected = case["expected_behavior"]
    criteria = case["evaluation_criteria"]

    # Check must_include (correctness)
    must_include = expected["must_include"]
    included = []
    missing = []
    for term in must_include:
        if re.search(re.escape(term), response, re.IGNORECASE):
            included.append(term)
        else:
            missing.append(term)

    correctness_score = len(included) / len(must_include) if must_include else 1.0

    # Check must_not_include (no hallucinations/errors)
    must_not = expected.get("must_not_include", [])
    violations = []
    for term in must_not:
        if re.search(re.escape(term), response, re.IGNORECASE):
            violations.append(term)

    violation_penalty = len(violations) * 0.2  # -20% per violation

    # Completeness: response length relative to ideal
    ideal = expected.get("ideal_response", "")
    if ideal:
        # Penalize very short responses, reward comprehensive ones
        response_len = len(response)
        ideal_len = len(ideal)
        if response_len < ideal_len * 0.3:
            completeness_score = 0.3
        elif response_len < ideal_len * 0.6:
            completeness_score = 0.6
        else:
            completeness_score = min(1.0, response_len / (ideal_len * 1.5))
    else:
        completeness_score = 0.5  # Default if no ideal

    # Clarity: simple heuristic (not too verbose, has structure)
    clarity_score = 1.0
    if len(response) > 2000:  # Too verbose
        clarity_score = 0.7
    if len(response) < 50:  # Too terse
        clarity_score = 0.5

    # Calculate weighted score
    raw_score = (
        correctness_score * criteria["correctness_weight"] +
        completeness_score * criteria["completeness_weight"] +
        clarity_score * criteria["clarity_weight"]
    )

    # Apply violation penalty
    final_score = max(0, raw_score - violation_penalty)

    return {
        "correctness": correctness_score,
        "completeness": completeness_score,
        "clarity": clarity_score,
        "final_score": round(final_score, 3),
        "included_terms": included,
        "missing_terms": missing,
        "violations": violations
    }


def run_single_case(model: str, case: dict, verbose: bool = False) -> dict:
    """Run a single benchmark case against a model."""
    # Build prompt
    prompt = case["input"]["prompt"]
    context = case["input"]["context"]
    if context:
        prompt = f"{prompt}\n\n```\n{context}\n```"

    # Query model
    response, latency = query_ollama(model, prompt)

    # Evaluate
    evaluation = evaluate_response(response, case)
    evaluation["latency_seconds"] = round(latency, 2)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Case: {case['id']} ({case['difficulty']})")
        print(f"Score: {evaluation['final_score']:.2f} | Latency: {latency:.1f}s")
        if evaluation["missing_terms"]:
            print(f"Missing: {evaluation['missing_terms']}")
        if evaluation["violations"]:
            print(f"Violations: {evaluation['violations']}")

    return {
        "case_id": case["id"],
        "category": case["category"],
        "difficulty": case["difficulty"],
        "response": response[:500] + "..." if len(response) > 500 else response,
        "evaluation": evaluation
    }


def load_cases(category: Optional[str] = None) -> list:
    """Load benchmark cases, optionally filtered by category."""
    if category:
        case_file = CASES_PATH / f"{category}.json"
        if not case_file.exists():
            raise FileNotFoundError(f"Category not found: {category}")
        with open(case_file) as f:
            return json.load(f)
    else:
        all_file = CASES_PATH / "all_cases.json"
        with open(all_file) as f:
            return json.load(f)


def run_benchmark(model: str, category: Optional[str] = None, verbose: bool = False) -> dict:
    """Run full benchmark against a model."""
    print(f"\n{'='*60}")
    print(f"FABRIK EVAL BENCHMARK")
    print(f"Model: {model}")
    print(f"Category: {category or 'ALL'}")
    print(f"{'='*60}")

    cases = load_cases(category)
    results = []

    for i, case in enumerate(cases):
        print(f"\r[{i+1}/{len(cases)}] Running {case['id']}...", end="", flush=True)
        result = run_single_case(model, case, verbose)
        results.append(result)

    print("\r" + " "*60, end="\r")  # Clear line

    # Calculate aggregates
    scores_by_category = {}
    scores_by_difficulty = {"easy": [], "medium": [], "hard": []}
    all_scores = []
    total_latency = 0

    for r in results:
        score = r["evaluation"]["final_score"]
        latency = r["evaluation"]["latency_seconds"]
        cat = r["category"]
        diff = r["difficulty"]

        all_scores.append(score)
        total_latency += latency
        scores_by_difficulty[diff].append(score)

        if cat not in scores_by_category:
            scores_by_category[cat] = []
        scores_by_category[cat].append(score)

    # Build summary
    summary = {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "total_cases": len(cases),
        "overall_score": round(sum(all_scores) / len(all_scores), 3),
        "avg_latency_seconds": round(total_latency / len(cases), 2),
        "by_category": {
            cat: round(sum(scores) / len(scores), 3)
            for cat, scores in scores_by_category.items()
        },
        "by_difficulty": {
            diff: round(sum(scores) / len(scores), 3) if scores else 0
            for diff, scores in scores_by_difficulty.items()
        }
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Overall Score: {summary['overall_score']:.3f}")
    print(f"Avg Latency: {summary['avg_latency_seconds']:.1f}s")
    print(f"\nBy Category:")
    for cat, score in summary["by_category"].items():
        print(f"  {cat}: {score:.3f}")
    print(f"\nBy Difficulty:")
    for diff, score in summary["by_difficulty"].items():
        print(f"  {diff}: {score:.3f}")

    return {
        "summary": summary,
        "results": results
    }


def save_results(benchmark_results: dict, model: str):
    """Save benchmark results to file."""
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = model.replace(":", "-").replace("/", "-")
    filename = f"{model_safe}_{timestamp}.json"

    filepath = RESULTS_PATH / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {filepath}")
    return filepath


def compare_models(models: list[str], category: Optional[str] = None) -> dict:
    """Run benchmark on multiple models and compare."""
    print(f"\n{'='*60}")
    print(f"COMPARATIVE BENCHMARK")
    print(f"Models: {', '.join(models)}")
    print(f"{'='*60}")

    all_results = {}
    for model in models:
        print(f"\n>>> Running benchmark for: {model}")
        benchmark = run_benchmark(model, category, verbose=False)
        all_results[model] = benchmark
        save_results(benchmark, model)

    # Print comparison table
    print(f"\n{'='*60}")
    print(f"COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Overall':<10} {'Latency':<10}")
    print("-" * 45)
    for model, result in all_results.items():
        s = result["summary"]
        print(f"{model:<25} {s['overall_score']:<10.3f} {s['avg_latency_seconds']:<10.1f}s")

    # Category breakdown
    categories = list(all_results[models[0]]["summary"]["by_category"].keys())
    print(f"\n{'Category Breakdown':}")
    header = f"{'Category':<15}" + "".join(f"{m:<12}" for m in models)
    print(header)
    print("-" * len(header))
    for cat in categories:
        row = f"{cat:<15}"
        for model in models:
            score = all_results[model]["summary"]["by_category"].get(cat, 0)
            row += f"{score:<12.3f}"
        print(row)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run Fabrik Eval Benchmark")
    parser.add_argument("--model", "-m", help="Model to evaluate (e.g., qwen2.5-coder:7b)")
    parser.add_argument("--category", "-c", help="Category filter (e.g., code-review)")
    parser.add_argument("--compare", help="Comma-separated models to compare")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--list-categories", action="store_true", help="List available categories")

    args = parser.parse_args()

    if args.list_categories:
        print("Available categories:")
        for f in CASES_PATH.glob("*.json"):
            if f.name != "all_cases.json":
                print(f"  - {f.stem}")
        return

    if args.compare:
        models = [m.strip() for m in args.compare.split(",")]
        compare_models(models, args.category)
    elif args.model:
        result = run_benchmark(args.model, args.category, args.verbose)
        save_results(result, args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
