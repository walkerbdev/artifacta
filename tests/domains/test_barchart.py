"""
End-to-end tests for BarChart primitive
Categorical data visualization for model comparison and metrics

Run with: pytest tests/domains/test_barchart.py -v
"""

import time

import pytest

import artifacta as ds


@pytest.mark.e2e
def test_model_performance_comparison():
    """Test BarChart: Compare multiple models across different metrics"""

    ds.init(project="model-comparison", name="multi-model-eval", config={"dataset": "ImageNet"})
    run = ds.get_run()

    # Model performance metrics (accuracy, precision, recall, f1)
    models = ["ResNet-50", "EfficientNet-B0", "ViT-Base", "MobileNetV2"]
    metrics = {
        "accuracy": [0.85, 0.88, 0.90, 0.82],
        "precision": [0.83, 0.86, 0.89, 0.80],
        "recall": [0.84, 0.87, 0.88, 0.81],
        "f1_score": [0.835, 0.865, 0.885, 0.805],
    }

    # Log as grouped bar chart
    run.log(
        "model_performance",
        ds.BarChart(
            categories=models,
            groups=metrics,
            x_label="Model Architecture",
            y_label="Score",
            stacked=False,
            horizontal=False,
        ),
        section="evaluation",
    )

    # Log same data as stacked bar chart
    run.log(
        "model_performance_stacked",
        ds.BarChart(
            categories=models,
            groups=metrics,
            x_label="Model Architecture",
            y_label="Total Score",
            stacked=True,
            horizontal=False,
        ),
        section="evaluation",
    )

    time.sleep(0.5)


@pytest.mark.e2e
def test_sales_by_region():
    """Test BarChart: Business analytics - Sales across regions and quarters"""

    ds.init(project="sales-analytics", name="q4-2024-report", config={"year": 2024})
    run = ds.get_run()

    # Quarterly sales by region (in millions)
    regions = ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East"]
    quarters = {
        "Q1": [45.2, 38.5, 52.3, 12.8, 8.4],
        "Q2": [48.6, 41.2, 58.1, 14.2, 9.1],
        "Q3": [52.1, 44.8, 63.5, 15.9, 10.3],
        "Q4": [58.3, 49.2, 71.2, 18.1, 11.8],
    }

    # Grouped bar chart showing quarters side-by-side
    run.log(
        "quarterly_sales",
        ds.BarChart(
            categories=regions,
            groups=quarters,
            x_label="Region",
            y_label="Revenue (Millions USD)",
            stacked=False,
        ),
        section="revenue",
    )

    # Stacked bar chart showing total revenue per region
    run.log(
        "total_sales_by_region",
        ds.BarChart(
            categories=regions,
            groups=quarters,
            x_label="Region",
            y_label="Total Annual Revenue (Millions USD)",
            stacked=True,
        ),
        section="revenue",
    )

    time.sleep(0.5)


@pytest.mark.e2e
def test_ab_test_conversion_rates():
    """Test BarChart: A/B testing results across variants"""

    ds.init(project="ab-testing", name="homepage-redesign-test", config={"test_duration_days": 14})
    run = ds.get_run()

    # Conversion rates for different user segments
    variants = ["Control", "Variant A", "Variant B", "Variant C"]
    segments = {
        "mobile_users": [0.042, 0.048, 0.051, 0.045],
        "desktop_users": [0.068, 0.075, 0.082, 0.071],
        "tablet_users": [0.055, 0.061, 0.064, 0.058],
    }

    run.log(
        "conversion_by_segment",
        ds.BarChart(
            categories=variants,
            groups=segments,
            x_label="Test Variant",
            y_label="Conversion Rate",
            stacked=False,
        ),
        section="results",
    )

    time.sleep(0.5)


@pytest.mark.e2e
def test_feature_importance():
    """Test BarChart: Feature importance from ML model"""

    ds.init(
        project="feature-analysis", name="random-forest-importance", config={"n_estimators": 100}
    )
    run = ds.get_run()

    # Feature importance scores
    features = [
        "age",
        "income",
        "credit_score",
        "employment_years",
        "debt_ratio",
        "num_accounts",
        "payment_history",
        "loan_amount",
    ]

    # Compare importance across different models
    importance_scores = {
        "random_forest": [0.15, 0.22, 0.18, 0.12, 0.08, 0.05, 0.13, 0.07],
        "gradient_boosting": [0.14, 0.24, 0.19, 0.11, 0.09, 0.04, 0.12, 0.07],
        "xgboost": [0.16, 0.23, 0.17, 0.13, 0.07, 0.06, 0.11, 0.07],
    }

    run.log(
        "feature_importance",
        ds.BarChart(
            categories=features,
            groups=importance_scores,
            x_label="Feature",
            y_label="Importance Score",
            stacked=False,
        ),
        section="analysis",
    )

    time.sleep(0.5)


@pytest.mark.e2e
def test_horizontal_bars():
    """Test BarChart: Horizontal orientation for long category names"""

    ds.init(project="survey-results", name="customer-satisfaction", config={"responses": 1250})
    run = ds.get_run()

    # Customer satisfaction across different service areas
    services = [
        "Product Quality",
        "Customer Support Response Time",
        "Shipping & Delivery",
        "Website User Experience",
        "Pricing & Value",
        "Return & Refund Process",
    ]

    ratings = {"satisfaction_score": [4.5, 4.2, 4.7, 4.1, 3.8, 4.3]}

    run.log(
        "satisfaction_ratings",
        ds.BarChart(
            categories=services,
            groups=ratings,
            x_label="Rating (out of 5)",
            y_label="Service Category",
            stacked=False,
            horizontal=True,
        ),
        section="feedback",
    )

    time.sleep(0.5)
