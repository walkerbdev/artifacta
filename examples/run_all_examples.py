"""
Run All Artifacta Examples
===========================

This script runs all example files in sequence and reports the results.
Useful for:
- Validating that all examples work correctly
- Regression testing after code changes
- Quick smoke test before releases

Usage:
    python examples/run_all_examples.py                    # Run all examples
    python examples/run_all_examples.py --category core    # Run only core examples
    python examples/run_all_examples.py --fast             # Skip long-running examples

Categories:
    - core: Basic Artifacta concepts and primitives
    - ml_frameworks: Machine learning framework integrations
    - domain_specific: Domain-specific use cases
    - all: All examples (default)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Define all examples with metadata
EXAMPLES = [
    # Core examples
    {
        "path": "core/01_basic_tracking.py",
        "name": "Basic Tracking",
        "category": "core",
        "duration": "fast",
        "description": "Minimal hello world example",
    },
    {
        "path": "core/02_all_primitives.py",
        "name": "All Primitives Demo",
        "category": "core",
        "duration": "fast",
        "description": "Showcase all 7 data primitives",
    },
    # ML Framework examples
    {
        "path": "ml_frameworks/sklearn_classification.py",
        "name": "Sklearn Classification",
        "category": "ml_frameworks",
        "duration": "medium",
        "description": "Sklearn with ROC/PR curves",
    },
    {
        "path": "ml_frameworks/xgboost_regression.py",
        "name": "XGBoost Regression",
        "category": "ml_frameworks",
        "duration": "medium",
        "description": "XGBoost with feature importance",
    },
    {
        "path": "ml_frameworks/pytorch_mnist.py",
        "name": "PyTorch MNIST",
        "category": "ml_frameworks",
        "duration": "slow",
        "description": "PyTorch Lightning with autolog",
    },
    {
        "path": "ml_frameworks/tensorflow_regression.py",
        "name": "TensorFlow Regression",
        "category": "ml_frameworks",
        "duration": "slow",
        "description": "TensorFlow/Keras with autolog",
    },
    # Domain-specific examples
    {
        "path": "domain_specific/ab_testing_experiment.py",
        "name": "A/B Testing",
        "category": "domain_specific",
        "duration": "fast",
        "description": "Domain-agnostic A/B test tracking",
    },
    {
        "path": "domain_specific/protein_expression.py",
        "name": "Protein Expression",
        "category": "domain_specific",
        "duration": "fast",
        "description": "Wet lab experiment optimization",
    },
]


def run_example(example_path: Path) -> dict:
    """Run a single example and return results.

    Args:
        example_path: Path to the example file

    Returns:
        dict with keys: success (bool), duration (float), output (str), error (str)
    """
    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(example_path)],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        duration = time.time() - start_time
        success = result.returncode == 0

        return {
            "success": success,
            "duration": duration,
            "output": result.stdout,
            "error": result.stderr if not success else None,
        }

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return {
            "success": False,
            "duration": duration,
            "output": "",
            "error": "Example timed out after 5 minutes",
        }

    except Exception as e:
        duration = time.time() - start_time
        return {
            "success": False,
            "duration": duration,
            "output": "",
            "error": str(e),
        }


def print_header():
    """Print script header."""
    print("=" * 80)
    print("Artifacta - Run All Examples")
    print("=" * 80)
    print()


def print_example_header(idx: int, total: int, example: dict):
    """Print example execution header."""
    print(f"\n[{idx}/{total}] {example['name']}")
    print(f"  File: {example['path']}")
    print(f"  Description: {example['description']}")
    print("  Running... ", end="", flush=True)


def print_result(result: dict):
    """Print example result."""
    if result["success"]:
        print(f"PASSED ({result['duration']:.1f}s)")
    else:
        print(f"FAILED ({result['duration']:.1f}s)")
        if result["error"]:
            print("\n  Error:")
            for line in result["error"].split("\n")[:10]:  # Show first 10 lines
                print(f"    {line}")


def print_summary(results: list[dict], examples: list[dict]):
    """Print final summary table."""
    passed = sum(1 for r in results if r["success"])
    failed = sum(1 for r in results if not r["success"])
    total_duration = sum(r["duration"] for r in results)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"\nTotal: {len(results)} examples")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"⏱  Total time: {total_duration:.1f}s")

    if failed > 0:
        print("\nFailed examples:")
        for example, result in zip(examples, results):
            if not result["success"]:
                print(f"  {example['name']} ({example['path']})")

    print("\n" + "=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run all Artifacta examples")
    parser.add_argument(
        "--category",
        choices=["core", "ml_frameworks", "domain_specific", "all"],
        default="all",
        help="Category of examples to run",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip slow examples (only run fast/medium duration)",
    )

    args = parser.parse_args()

    # Filter examples based on arguments
    examples_to_run = EXAMPLES

    if args.category != "all":
        examples_to_run = [e for e in examples_to_run if e["category"] == args.category]

    if args.fast:
        examples_to_run = [e for e in examples_to_run if e["duration"] != "slow"]

    if not examples_to_run:
        print("No examples match the specified filters.")
        return 1

    # Print header
    print_header()
    print(f"Running {len(examples_to_run)} examples")
    if args.category != "all":
        print(f"Category: {args.category}")
    if args.fast:
        print("Mode: Fast (skipping slow examples)")
    print()

    # Get base directory
    examples_dir = Path(__file__).parent

    # Run each example
    results = []
    for idx, example in enumerate(examples_to_run, 1):
        example_path = examples_dir / example["path"]

        if not example_path.exists():
            print(f"\n[{idx}/{len(examples_to_run)}] {example['name']}")
            print(f"  SKIPPED - File not found: {example_path}")
            results.append({
                "success": False,
                "duration": 0,
                "output": "",
                "error": f"File not found: {example_path}",
            })
            continue

        print_example_header(idx, len(examples_to_run), example)
        result = run_example(example_path)
        results.append(result)
        print_result(result)

    # Print summary
    print_summary(results, examples_to_run)

    # Exit with error code if any examples failed
    return 1 if any(not r["success"] for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
