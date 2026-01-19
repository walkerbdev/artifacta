import pandas as pd
from scipy.stats import chi2_contingency


class ABTestAnalyzer:
    def __init__(self, traffic_split=0.5, min_sample_size=1000, significance_level=0.05):
        self.traffic_split = traffic_split
        self.min_sample_size = min_sample_size
        self.significance_level = significance_level

    def analyze_experiment(self, control_data, variant_data):
        """Analyze A/B test results using chi-square test"""
        # Statistical significance test
        control_conversions = sum(control_data["converted"])
        variant_conversions = sum(variant_data["converted"])

        control_total = len(control_data)
        variant_total = len(variant_data)

        # Chi-square test for independence
        contingency_table = [
            [control_conversions, control_total - control_conversions],
            [variant_conversions, variant_total - variant_conversions],
        ]
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        # Calculate conversion rates
        control_rate = control_conversions / control_total
        variant_rate = variant_conversions / variant_total

        # Relative improvement
        relative_improvement = (variant_rate - control_rate) / control_rate * 100

        return {
            "control_rate": control_rate,
            "variant_rate": variant_rate,
            "p_value": p_value,
            "statistically_significant": p_value < self.significance_level,
            "relative_improvement": relative_improvement,
        }


# Example usage
if __name__ == "__main__":
    analyzer = ABTestAnalyzer(traffic_split=0.5, min_sample_size=1000, significance_level=0.05)

    # Load your experiment data
    control_df = pd.read_csv("control_data.csv")
    variant_df = pd.read_csv("variant_data.csv")

    results = analyzer.analyze_experiment(control_df, variant_df)
    print(f"Conversion rate lift: {results['relative_improvement']:.2f}%")
    print(f"Statistical significance: {results['statistically_significant']}")
