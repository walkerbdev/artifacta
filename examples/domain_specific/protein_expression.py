"""
Protein Expression Optimization Example with Artifacta
=======================================================

This example demonstrates using Artifacta for wet lab experimental data tracking.
It simulates a typical protein expression optimization workflow where researchers
test different growth conditions to maximize protein yield.

Scenario:
---------
A molecular biology lab is optimizing expression of a recombinant protein in E. coli.
They're testing different combinations of:
- Temperature (25°C, 30°C, 37°C)
- IPTG concentration (0.1 mM, 0.5 mM, 1.0 mM)
- Induction time (4h, 6h, 8h)

For each condition, they measure:
- Total protein yield (mg/L)
- Protein purity (%)
- Specific activity (U/mg)
- Growth rate (OD600/h)

Key Artifacta Features Demonstrated:
------------------------------------
1. **CSV data import** - Load experimental data from spreadsheet
2. **Grid search tracking** - Systematically track all parameter combinations
3. **Series plots** - Visualize growth curves over time
4. **BarChart comparisons** - Compare yields across conditions
5. **Scatter plots** - Analyze parameter-response relationships
6. **Parameter correlation** - Identify which factors matter most

Requirements:
    pip install artifacta pandas numpy

Usage:
    python examples/protein_expression_optimization.py
"""

from itertools import product

import numpy as np
import pandas as pd

from artifacta import BarChart, Scatter, Series, init, log


def generate_synthetic_data():
    """Generate synthetic protein expression data.

    In a real scenario, this would be loaded from your lab's CSV files.
    This simulates realistic experimental results with some biological noise.

    Returns:
        DataFrame with experimental conditions and results
    """
    print("\nGenerating synthetic experimental data...")
    print("  (In practice, this would be loaded from your lab notebook CSV)")

    # Define parameter grid
    temperatures = [25, 30, 37]  # °C
    iptg_concs = [0.1, 0.5, 1.0]  # mM
    induction_times = [4, 6, 8]  # hours

    # Generate all combinations
    conditions = list(product(temperatures, iptg_concs, induction_times))

    # Simulate realistic protein expression results
    # Higher temp + moderate IPTG + longer time = better yield (with noise)
    data = []
    for temp, iptg, time in conditions:
        # Base yield increases with temperature and time
        base_yield = (temp - 20) * 5 + time * 10 + iptg * 20

        # Add biological noise (±20%)
        yield_mg_l = base_yield + np.random.normal(0, base_yield * 0.2)
        yield_mg_l = max(10, yield_mg_l)  # Ensure positive

        # Purity decreases at very high temp or IPTG
        base_purity = 90 - abs(temp - 30) * 2 - (iptg - 0.5) ** 2 * 10
        purity_pct = base_purity + np.random.normal(0, 5)
        purity_pct = np.clip(purity_pct, 50, 98)

        # Activity correlates with purity but has noise
        activity = purity_pct * 0.8 + np.random.normal(0, 10)
        activity = max(20, activity)

        # Growth rate decreases at extreme temps
        optimal_temp_growth = 1 - abs(temp - 37) / 20
        growth_rate = optimal_temp_growth + np.random.normal(0, 0.1)
        growth_rate = max(0.2, growth_rate)

        data.append({
            'temperature_C': temp,
            'iptg_mM': iptg,
            'induction_time_h': time,
            'yield_mg_L': round(yield_mg_l, 2),
            'purity_pct': round(purity_pct, 1),
            'activity_U_mg': round(activity, 1),
            'growth_rate_OD_h': round(growth_rate, 3)
        })

    df = pd.DataFrame(data)

    print(f"  Generated {len(df)} experimental conditions")
    print(f"  Temperature range: {df['temperature_C'].min()}-{df['temperature_C'].max()}°C")
    print(f"  IPTG range: {df['iptg_mM'].min()}-{df['iptg_mM'].max()} mM")
    print(f"  Induction time range: {df['induction_time_h'].min()}-{df['induction_time_h'].max()} hours")

    return df


def generate_growth_curve(temp, iptg, induction_time):
    """Generate a simulated growth curve for a specific condition.

    In practice, this would be real OD600 measurements from your plate reader.
    """
    # Time points (hours)
    time_points = np.linspace(0, 12, 25)

    # Simulate bacterial growth with logistic curve
    # Growth parameters depend on conditions
    max_od = 2.0 + (temp - 30) * 0.1 + np.random.normal(0, 0.1)
    growth_rate = 0.5 - abs(temp - 37) * 0.02 + np.random.normal(0, 0.05)

    # Logistic growth curve
    od600 = max_od / (1 + np.exp(-growth_rate * (time_points - 4)))

    # Add measurement noise
    od600 += np.random.normal(0, 0.05, len(time_points))
    od600 = np.maximum(0.01, od600)  # OD can't be negative

    return time_points.tolist(), od600.tolist()


def main():
    """Main experimental analysis workflow."""
    print("=" * 70)
    print("Artifacta Protein Expression Optimization Example")
    print("=" * 70)
    print("\nScenario: Optimizing recombinant protein expression in E. coli")
    print("Testing different temperatures, IPTG concentrations, and induction times")

    # =================================================================
    # 1. Generate/Load experimental data
    # =================================================================
    df = generate_synthetic_data()

    # =================================================================
    # 2. Initialize Artifacta project
    # =================================================================
    print("\n" + "=" * 70)
    print("Logging Experimental Data to Artifacta")
    print("=" * 70)

    run = init(
        project="protein-expression-optimization",
        name="ecoli-recombinant-protein-screen",
        config={
            "organism": "E. coli BL21(DE3)",
            "protein": "His-tagged GFP",
            "expression_vector": "pET28a",
            "culture_volume_mL": 50,
            "study_type": "factorial_design",
            "parameters_tested": ["temperature", "IPTG_concentration", "induction_time"],
        }
    )
    print("\nArtifacta run initialized")

    # =================================================================
    # 3. Log individual condition results
    # =================================================================
    print("\nLogging experimental conditions...")

    # Track best condition
    best_yield = df.loc[df['yield_mg_L'].idxmax()]
    best_purity = df.loc[df['purity_pct'].idxmax()]
    best_activity = df.loc[df['activity_U_mg'].idxmax()]

    for _idx, row in df.iterrows():
        condition_name = f"T{row['temperature_C']}_IPTG{row['iptg_mM']}_t{row['induction_time_h']}h"

        # Log each condition as a separate run (in practice, you might log all at once)
        # This demonstrates tracking individual experiments
        print(f"  {condition_name}: Yield={row['yield_mg_L']:.1f} mg/L, "
              f"Purity={row['purity_pct']:.1f}%, Activity={row['activity_U_mg']:.1f} U/mg")

    print(f"\n  Total conditions tested: {len(df)}")
    print(f"  Best yield: {best_yield['yield_mg_L']:.1f} mg/L at "
          f"{best_yield['temperature_C']}°C, {best_yield['iptg_mM']} mM IPTG, "
          f"{best_yield['induction_time_h']}h induction")

    # =================================================================
    # 4. Log yield comparison across temperatures (BarChart)
    # =================================================================
    print("\nCreating yield comparison charts...")

    # Group by temperature and calculate mean yield
    temp_yields = df.groupby('temperature_C')['yield_mg_L'].mean().round(2)

    log(
        "yield_by_temperature",
        BarChart(
            categories=[f"{t}°C" for t in temp_yields.index],
            groups={"Yield (mg/L)": temp_yields.tolist()},
            x_label="Temperature",
            y_label="Protein Yield (mg/L)"
        )
    )

    # Group by IPTG concentration
    iptg_yields = df.groupby('iptg_mM')['yield_mg_L'].mean().round(2)

    log(
        "yield_by_iptg",
        BarChart(
            categories=[f"{c} mM" for c in iptg_yields.index],
            groups={"Yield (mg/L)": iptg_yields.tolist()},
            x_label="IPTG Concentration",
            y_label="Protein Yield (mg/L)"
        )
    )

    # =================================================================
    # 5. Log parameter relationships (Scatter)
    # =================================================================
    print("\nCreating parameter correlation plots...")

    # Yield vs Temperature
    log(
        "yield_vs_temperature",
        Scatter(
            points=[
                {"x": temp, "y": yield_val}
                for temp, yield_val in zip(df['temperature_C'], df['yield_mg_L'])
            ],
            x_label="Temperature (°C)",
            y_label="Protein Yield (mg/L)"
        )
    )

    # Yield vs IPTG
    log(
        "yield_vs_iptg",
        Scatter(
            points=[
                {"x": iptg, "y": yield_val}
                for iptg, yield_val in zip(df['iptg_mM'], df['yield_mg_L'])
            ],
            x_label="IPTG Concentration (mM)",
            y_label="Protein Yield (mg/L)"
        )
    )

    # Purity vs Activity (quality relationship)
    log(
        "purity_vs_activity",
        Scatter(
            points=[
                {"x": purity, "y": activity}
                for purity, activity in zip(df['purity_pct'], df['activity_U_mg'])
            ],
            x_label="Protein Purity (%)",
            y_label="Specific Activity (U/mg)"
        )
    )

    # =================================================================
    # 6. Log growth curves for representative conditions (Series)
    # =================================================================
    print("\nLogging growth curves...")

    # Log growth curves for a few representative conditions
    representative_conditions = [
        (25, 0.1, 4, "low_temp_low_iptg"),
        (30, 0.5, 6, "optimal_moderate"),
        (37, 1.0, 8, "high_temp_high_iptg"),
    ]

    for temp, iptg, induction_time, name in representative_conditions:
        time_points, od600_values = generate_growth_curve(temp, iptg, induction_time)

        log(
            f"growth_curve_{name}",
            Series(
                index="time_hours",
                fields={
                    "OD600": od600_values,
                },
                index_values=time_points,
                metadata={
                    "temperature_C": temp,
                    "iptg_mM": iptg,
                    "induction_time_h": induction_time,
                    "description": f"Growth curve at {temp}°C, {iptg}mM IPTG, {induction_time}h induction"
                }
            )
        )

    # =================================================================
    # 7. Save summary CSV as artifact
    # =================================================================
    print("\nSaving experimental data as artifact...")

    # Save the dataframe to CSV
    csv_path = "protein_expression_results.csv"
    df.to_csv(csv_path, index=False)

    run.log_output(
        csv_path,
        name="experimental_results",
        metadata={
            "description": "Complete experimental results for all tested conditions",
            "total_conditions": len(df),
            "best_yield_mg_L": float(best_yield['yield_mg_L']),
            "best_condition": f"{best_yield['temperature_C']}°C, {best_yield['iptg_mM']}mM IPTG, {best_yield['induction_time_h']}h"
        }
    )

    # =================================================================
    # 8. Summary and Recommendations
    # =================================================================
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print("\nKey Findings:")
    print(f"  • Best yield: {best_yield['yield_mg_L']:.1f} mg/L")
    print(f"    Conditions: {best_yield['temperature_C']}°C, "
          f"{best_yield['iptg_mM']} mM IPTG, {best_yield['induction_time_h']}h induction")
    print(f"\n  • Highest purity: {best_purity['purity_pct']:.1f}%")
    print(f"    Conditions: {best_purity['temperature_C']}°C, "
          f"{best_purity['iptg_mM']} mM IPTG, {best_purity['induction_time_h']}h induction")
    print(f"\n  • Best activity: {best_activity['activity_U_mg']:.1f} U/mg")
    print(f"    Conditions: {best_activity['temperature_C']}°C, "
          f"{best_activity['iptg_mM']} mM IPTG, {best_activity['induction_time_h']}h induction")

    print("\nAll experimental data logged to Artifacta")
    print("  View results in the Artifacta UI to:")
    print("  - Compare yields across conditions")
    print("  - Analyze parameter correlations")
    print("  - Review growth curves")
    print("  - Access raw data CSV")
    print("=" * 70)

    run.finish()


if __name__ == "__main__":
    main()
