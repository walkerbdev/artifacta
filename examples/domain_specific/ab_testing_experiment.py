"""
A/B Testing Experiment Simulation with Artifacta
=================================================

This example demonstrates that Artifacta works for ANY experiment tracking, not just ML!

This example shows:
1. **Domain-agnostic tracking** - Track A/B tests, marketing experiments, etc.
2. **Parameter sweeps** - Run multiple experiments with different configurations
3. **Distribution analysis** via Distribution - Compare conversion rates by variant
4. **Time series tracking** via Series - Monitor cumulative metrics over time
5. **Category comparison** via BarChart - Visualize performance across variants

Scenario:
We're testing different button colors on an e-commerce checkout page to see which
drives more conversions. We'll run multiple experiments with different sample sizes
and traffic splits to demonstrate parameter sweeps.

Key Artifacta Features Demonstrated:
- init() - Initialize experiment run with config (NOT ML-specific!)
- Distribution - Log conversion rates by variant (grouped data)
- Series - Log cumulative conversions over time
- BarChart - Compare final metrics across variants
- Parameter sweeps - Run multiple experiments systematically

Requirements:
    pip install artifacta numpy

Usage:
    python examples/ab_testing_experiment.py
"""

import time
from typing import Dict, List

import numpy as np

from artifacta import BarChart, Distribution, Series, init, log


def simulate_button_test(
    n_control: int,
    n_variant_a: int,
    n_variant_b: int,
    control_rate: float = 0.05,
    variant_a_rate: float = 0.062,
    variant_b_rate: float = 0.058,
    random_state: int = 42,
) -> Dict:
    """Simulate A/B test for button color experiment.

    Simulates user conversions for three button variants:
    - Control: Blue button (baseline)
    - Variant A: Green button (best performer)
    - Variant B: Red button (middle performer)

    Args:
        n_control: Number of users shown control button
        n_variant_a: Number of users shown variant A
        n_variant_b: Number of users shown variant B
        control_rate: True conversion rate for control
        variant_a_rate: True conversion rate for variant A
        variant_b_rate: True conversion rate for variant B
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with conversion data and statistics
    """
    np.random.seed(random_state)

    # Simulate conversions (1 = converted, 0 = didn't convert)
    # We use Bernoulli trials (coin flips) with different success probabilities
    control_conversions = np.random.binomial(1, control_rate, n_control)
    variant_a_conversions = np.random.binomial(1, variant_a_rate, n_variant_a)
    variant_b_conversions = np.random.binomial(1, variant_b_rate, n_variant_b)

    # Calculate observed conversion rates
    control_observed_rate = control_conversions.mean()
    variant_a_observed_rate = variant_a_conversions.mean()
    variant_b_observed_rate = variant_b_conversions.mean()

    # For Distribution: we need individual values with group labels
    # Create conversion rate samples (simulating individual user outcomes as rates)
    control_samples = control_conversions.astype(float)
    variant_a_samples = variant_a_conversions.astype(float)
    variant_b_samples = variant_b_conversions.astype(float)

    return {
        "control": {
            "n_users": n_control,
            "n_conversions": int(control_conversions.sum()),
            "conversion_rate": control_observed_rate,
            "samples": control_samples,
        },
        "variant_a": {
            "n_users": n_variant_a,
            "n_conversions": int(variant_a_conversions.sum()),
            "conversion_rate": variant_a_observed_rate,
            "samples": variant_a_samples,
        },
        "variant_b": {
            "n_users": n_variant_b,
            "n_conversions": int(variant_b_conversions.sum()),
            "conversion_rate": variant_b_observed_rate,
            "samples": variant_b_samples,
        },
    }


def simulate_time_series(
    total_conversions: Dict[str, int], n_hours: int = 24, random_state: int = 42
) -> Dict[str, List[float]]:
    """Simulate cumulative conversions over time.

    Models how conversions accumulate hour by hour during the test.

    Args:
        total_conversions: Dict with total conversions per variant
        n_hours: Number of hours to simulate
        random_state: Random seed

    Returns:
        Dictionary with cumulative conversion arrays per variant
    """
    np.random.seed(random_state)

    def generate_cumulative(total: int, n_points: int) -> List[float]:
        """Generate cumulative sum that reaches total."""
        # Generate random increments that sum to total
        increments = np.random.dirichlet(np.ones(n_points)) * total
        cumulative = np.cumsum(increments)
        return cumulative.tolist()

    return {
        "control": generate_cumulative(total_conversions["control"], n_hours),
        "variant_a": generate_cumulative(total_conversions["variant_a"], n_hours),
        "variant_b": generate_cumulative(total_conversions["variant_b"], n_hours),
        "hours": list(range(1, n_hours + 1)),
    }


def calculate_statistical_significance(n1: int, conv1: int, n2: int, conv2: int) -> Dict:
    """Calculate statistical significance between two variants.

    Uses two-proportion z-test to determine if difference is significant.

    Args:
        n1: Sample size for variant 1
        conv1: Conversions for variant 1
        n2: Sample size for variant 2
        conv2: Conversions for variant 2

    Returns:
        Dictionary with significance results
    """
    p1 = conv1 / n1
    p2 = conv2 / n2

    # Pooled proportion
    p_pool = (conv1 + conv2) / (n1 + n2)

    # Standard error
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))

    # Z-score
    z_score = 0 if se == 0 else (p2 - p1) / se

    # P-value (two-tailed)
    from scipy import stats

    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    # Confidence level
    if p_value < 0.001:
        confidence = "99.9%"
        significant = True
    elif p_value < 0.01:
        confidence = "99%"
        significant = True
    elif p_value < 0.05:
        confidence = "95%"
        significant = True
    else:
        confidence = "Not significant"
        significant = False

    return {
        "p_value": p_value,
        "z_score": z_score,
        "confidence": confidence,
        "significant": significant,
        "lift": ((p2 - p1) / p1 * 100) if p1 > 0 else 0,
    }


def run_ab_test_experiment(config: Dict, seed: int = 42):
    """Run a single A/B test experiment with given configuration.

    Args:
        config: Configuration dictionary with test parameters
        seed: Random seed for reproducibility
    """
    print(f"\n  Traffic split: {config['traffic_split']}")
    print(f"  Sample size per variant: {config['sample_size_per_variant']}")
    print(f"  Test duration: {config['test_duration_hours']} hours")

    # Calculate sample sizes for each variant
    # Traffic is split equally among all variants
    n_control = config["sample_size_per_variant"]
    n_variant_a = config["sample_size_per_variant"]
    n_variant_b = config["sample_size_per_variant"]

    # Simulate the button test
    results = simulate_button_test(
        n_control=n_control,
        n_variant_a=n_variant_a,
        n_variant_b=n_variant_b,
        control_rate=0.05,  # Control: 5% conversion
        variant_a_rate=0.062,  # Variant A: 6.2% conversion (24% lift!)
        variant_b_rate=0.058,  # Variant B: 5.8% conversion (16% lift)
        random_state=seed,
    )

    print("\n  Results:")
    print(
        f"    Control (Blue):    {results['control']['conversion_rate']:.4f} "
        f"({results['control']['n_conversions']}/{results['control']['n_users']})"
    )
    print(
        f"    Variant A (Green): {results['variant_a']['conversion_rate']:.4f} "
        f"({results['variant_a']['n_conversions']}/{results['variant_a']['n_users']})"
    )
    print(
        f"    Variant B (Red):   {results['variant_b']['conversion_rate']:.4f} "
        f"({results['variant_b']['n_conversions']}/{results['variant_b']['n_users']})"
    )

    # =================================================================
    # 1. Log conversion rate distribution by variant
    #    Shows the distribution of conversion outcomes
    # =================================================================
    all_values = np.concatenate(
        [
            results["control"]["samples"],
            results["variant_a"]["samples"],
            results["variant_b"]["samples"],
        ]
    )

    all_groups = (
        ["Control (Blue)"] * len(results["control"]["samples"])
        + ["Variant A (Green)"] * len(results["variant_a"]["samples"])
        + ["Variant B (Red)"] * len(results["variant_b"]["samples"])
    )

    log(
        "conversion_by_variant",
        Distribution(
            values=all_values.tolist(),
            groups=all_groups,
            metadata={
                "description": "Binary conversion outcomes by button color",
                "control_rate": results["control"]["conversion_rate"],
                "variant_a_rate": results["variant_a"]["conversion_rate"],
                "variant_b_rate": results["variant_b"]["conversion_rate"],
            },
        ),
    )

    # =================================================================
    # 2. Log cumulative conversions over time
    #    Shows how conversions accumulate during the test
    # =================================================================
    total_conversions = {
        "control": results["control"]["n_conversions"],
        "variant_a": results["variant_a"]["n_conversions"],
        "variant_b": results["variant_b"]["n_conversions"],
    }

    time_series = simulate_time_series(
        total_conversions, n_hours=config["test_duration_hours"], random_state=seed
    )

    log(
        "cumulative_conversions",
        Series(
            index="hour",
            fields={
                "control": time_series["control"],
                "variant_a": time_series["variant_a"],
                "variant_b": time_series["variant_b"],
            },
            index_values=time_series["hours"],
            metadata={
                "description": "Cumulative conversions over test duration",
                "unit": "conversions",
            },
        ),
    )

    # =================================================================
    # 3. Calculate statistical significance
    # =================================================================
    sig_a = calculate_statistical_significance(
        n_control,
        results["control"]["n_conversions"],
        n_variant_a,
        results["variant_a"]["n_conversions"],
    )

    sig_b = calculate_statistical_significance(
        n_control,
        results["control"]["n_conversions"],
        n_variant_b,
        results["variant_b"]["n_conversions"],
    )

    print("\n  Statistical Significance:")
    print(f"    Variant A vs Control: {sig_a['confidence']} (lift: {sig_a['lift']:.1f}%)")
    print(f"    Variant B vs Control: {sig_b['confidence']} (lift: {sig_b['lift']:.1f}%)")

    # =================================================================
    # 4. Log comparison bar chart
    #    Shows side-by-side comparison of key metrics
    # =================================================================
    log(
        "variant_comparison",
        BarChart(
            categories=["Control (Blue)", "Variant A (Green)", "Variant B (Red)"],
            groups={
                "Conversion Rate (%)": [
                    results["control"]["conversion_rate"] * 100,
                    results["variant_a"]["conversion_rate"] * 100,
                    results["variant_b"]["conversion_rate"] * 100,
                ],
                "Total Conversions": [
                    float(results["control"]["n_conversions"]),
                    float(results["variant_a"]["n_conversions"]),
                    float(results["variant_b"]["n_conversions"]),
                ],
            },
            x_label="Button Variant",
            y_label="Value",
            metadata={
                "description": "Comparison of button variants",
                "winner": "Variant A (Green)" if sig_a["significant"] else "Inconclusive",
            },
        ),
    )

    # =================================================================
    # 5. Log final summary metrics as Series
    # =================================================================
    log(
        "summary_metrics",
        Series(
            index="variant",
            fields={
                "conversion_rate_pct": [
                    results["control"]["conversion_rate"] * 100,
                    results["variant_a"]["conversion_rate"] * 100,
                    results["variant_b"]["conversion_rate"] * 100,
                ],
                "sample_size": [
                    float(n_control),
                    float(n_variant_a),
                    float(n_variant_b),
                ],
            },
            index_values=["Control", "Variant_A", "Variant_B"],
        ),
    )


def main():
    """Main function - runs parameter sweep across different configurations."""
    print("=" * 70)
    print("Artifacta A/B Testing Example")
    print("Demonstrating: Artifacta works for ANY experiment, not just ML!")
    print("=" * 70)

    print("\nScenario:")
    print("  We're testing button colors on our e-commerce checkout page.")
    print("  - Control: Blue button (current baseline)")
    print("  - Variant A: Green button")
    print("  - Variant B: Red button")
    print("\n  Goal: Which button color drives the most conversions?")

    # =================================================================
    # Parameter sweep configuration
    #    Run multiple experiments with different settings
    # =================================================================
    base_config = {
        "experiment_type": "button_color_test",
        "page": "checkout",
        "metric": "conversion_rate",
        "test_duration_hours": 24,
    }

    # Different configurations to test
    param_variations = [
        {
            "name": "small_sample",
            "sample_size_per_variant": 500,
            "traffic_split": 0.33,
            "description": "Small sample size - may not be conclusive",
        },
        {
            "name": "medium_sample",
            "sample_size_per_variant": 2000,
            "traffic_split": 0.33,
            "description": "Medium sample size - better confidence",
        },
        {
            "name": "large_sample",
            "sample_size_per_variant": 5000,
            "traffic_split": 0.33,
            "description": "Large sample size - high confidence",
        },
    ]

    print(f"\nRunning parameter sweep with {len(param_variations)} configurations...")

    # =================================================================
    # Run experiments with different configurations
    # =================================================================
    for idx, variation in enumerate(param_variations):
        print("\n" + "=" * 70)
        print(f"Experiment {idx + 1}/{len(param_variations)}: {variation['name']}")
        print("=" * 70)
        print(f"  {variation['description']}")

        # Merge base config with variation
        config = {**base_config, **variation}

        # Initialize Artifacta run for this experiment
        run_name = f"button-test-{variation['name']}"
        init(project="ab-testing-demo", name=run_name, config=config)

        # Run the experiment
        run_ab_test_experiment(config, seed=42 + idx)

        # Small delay between runs
        time.sleep(0.3)

    # =================================================================
    # Final summary
    # =================================================================
    print("\n" + "=" * 70)
    print("Parameter Sweep Complete!")
    print("=" * 70)
    print("\nKey Findings:")
    print("  - Larger sample sizes provide more confident results")
    print("  - Variant A (Green) consistently shows best performance")
    print("  - Statistical significance increases with sample size")
    print("\nRecommendation:")
    print("  → Deploy Variant A (Green button) to production!")
    print("  → Expected conversion lift: ~24%")
    print("\nAll experiments logged to Artifacta")
    print("  View detailed results in the Artifacta UI!")
    print("  Compare all runs in the project view to see how sample size")
    print("  affects confidence and statistical significance.")
    print("=" * 70)


if __name__ == "__main__":
    # Import scipy.stats for significance testing
    try:
        from scipy import stats  # noqa: F401
    except ImportError:
        print("Error: scipy is not installed.")
        print("Please install it with: pip install scipy")
        print("Or run without statistical significance calculations.")
        exit(1)

    main()
