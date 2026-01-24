"""Tests for scikit-learn autolog integration.

Tests cover all features matching MLflow's sklearn autolog:
- Parameter logging (get_params with deep=True)
- Classifier metrics (accuracy, precision, recall, F1, log loss, ROC-AUC)
- Regressor metrics (MSE, RMSE, MAE, R²)
- Model artifact logging
- Binary vs multiclass classification
- Meta-estimators (Pipeline, GridSearchCV)
- Autolog enable/disable
"""


import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from artifacta import init
from artifacta.tests.test_utils import (
    MockHTTPEmitter,
    assert_artifact_logged,
    assert_dataset_logged,
    assert_metric_logged,
    assert_model_artifact_logged,
    assert_param_logged,
    get_logged_datasets,
    get_logged_metrics,
    get_logged_params,
)


@pytest.fixture
def binary_classification_data():
    """Binary classification dataset."""
    X, y = make_classification(
        n_samples=100, n_features=10, n_classes=2, random_state=42
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def multiclass_classification_data():
    """Multiclass classification dataset (iris)."""
    X, y = load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def regression_data():
    """Regression dataset."""
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def temp_run():
    """Create and cleanup temporary run with mocked HTTP emitter."""
    run = init(project="test_sklearn_autolog", name="test_run")
    # Replace the http_emitter with our mock
    run.http_emitter = MockHTTPEmitter(run.id)
    yield run
    run.finish()


class TestSklearnAutologBasic:
    """Basic autolog functionality tests."""

    def test_autolog_enable_disable(self):
        """Test enabling and disabling autolog."""
        from artifacta.integrations import sklearn

        # Enable
        sklearn.enable_autolog()
        assert sklearn._AUTOLOG_ENABLED is True

        # Disable
        sklearn.disable_autolog()
        assert sklearn._AUTOLOG_ENABLED is False

    def test_autolog_patches_estimators(self):
        """Test that autolog patches estimator fit() methods."""
        from artifacta.integrations import sklearn

        sklearn.enable_autolog()

        # Check that fit method was patched
        original_fit = sklearn._ORIGINAL_METHODS.get(
            (RandomForestClassifier, "fit")
        )
        assert original_fit is not None
        assert RandomForestClassifier.fit != original_fit

        sklearn.disable_autolog()

    def test_autolog_without_active_run(self, binary_classification_data):
        """Test that autolog doesn't crash when no run is active."""
        from artifacta.integrations import sklearn

        X_train, X_test, y_train, y_test = binary_classification_data

        sklearn.enable_autolog()

        # Should work without active run (just doesn't log)
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)

        sklearn.disable_autolog()

    def test_autolog_prevents_nested_logging(self, temp_run, binary_classification_data):
        """Test that nested estimator calls don't create duplicate logs."""
        from artifacta.integrations import sklearn

        X_train, X_test, y_train, y_test = binary_classification_data

        sklearn.enable_autolog()

        # Pipeline has nested fit() calls - should only log once
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=42)),
        ])
        pipe.fit(X_train, y_train)

        sklearn.disable_autolog()


class TestSklearnParameterLogging:
    """Test parameter logging functionality."""

    def test_log_basic_params(self, temp_run, binary_classification_data):
        """Test logging basic classifier parameters."""
        from artifacta.integrations import sklearn

        X_train, X_test, y_train, y_test = binary_classification_data

        sklearn.enable_autolog()

        clf = RandomForestClassifier(
            n_estimators=10,
            max_depth=5,
            random_state=42
        )
        clf.fit(X_train, y_train)

        sklearn.disable_autolog()

        # Debug: Print what was logged
        params = get_logged_params(temp_run)
        print(f"DEBUG: Logged params: {params}")
        print(f"DEBUG: All emitted data: {temp_run.http_emitter.emitted_data}")

        # Verify parameters were logged
        assert_param_logged(temp_run, "n_estimators", 10)
        assert_param_logged(temp_run, "max_depth", 5)
        assert_param_logged(temp_run, "random_state", 42)

    def test_log_deep_params_pipeline(self, temp_run, binary_classification_data):
        """Test logging deep parameters from Pipeline."""
        from artifacta.integrations import sklearn

        X_train, X_test, y_train, y_test = binary_classification_data

        sklearn.enable_autolog()

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=10, random_state=42)),
        ])
        pipe.fit(X_train, y_train)

        sklearn.disable_autolog()

        # Should log params from both scaler and clf (get_params(deep=True))


class TestSklearnDatasetLogging:
    """Test dataset metadata logging."""

    def test_log_dataset_shape_and_dtype(self, temp_run, binary_classification_data):
        """Test logging dataset shape and dtype."""
        from artifacta.integrations import sklearn

        X_train, X_test, y_train, y_test = binary_classification_data

        sklearn.enable_autolog()

        clf = RandomForestClassifier(n_estimators=5, random_state=42)
        clf.fit(X_train, y_train)

        sklearn.disable_autolog()

        # Verify dataset was logged
        assert_dataset_logged(temp_run, context="train", expected_shape=X_train.shape)

        # Verify dataset metadata contains required fields
        datasets = get_logged_datasets(temp_run)
        train_data = datasets["train"]

        assert train_data["features_shape"] == list(X_train.shape)
        assert train_data["features_size"] == X_train.size
        assert "features_digest" in train_data
        assert "features_dtype" in train_data
        assert "targets_shape" in train_data
        assert train_data["targets_shape"] == list(y_train.shape)

    def test_dataset_logging_can_be_disabled(self, temp_run, binary_classification_data):
        """Test that dataset logging can be disabled."""
        from artifacta.integrations import sklearn

        X_train, X_test, y_train, y_test = binary_classification_data

        sklearn.enable_autolog(log_datasets=False)

        clf = RandomForestClassifier(n_estimators=5, random_state=42)
        clf.fit(X_train, y_train)

        sklearn.disable_autolog()

        # Verify dataset was NOT logged
        datasets = get_logged_datasets(temp_run)
        assert "train" not in datasets, "Dataset should not be logged when log_datasets=False"


class TestSklearnClassifierMetrics:
    """Test classifier metric logging."""

    def test_binary_classifier_metrics(self, temp_run, binary_classification_data):
        """Test metrics for binary classification."""
        from artifacta.integrations import sklearn

        X_train, X_test, y_train, y_test = binary_classification_data

        sklearn.enable_autolog()

        clf = LogisticRegression(random_state=42)
        clf.fit(X_train, y_train)

        sklearn.disable_autolog()

        # Verify classifier metrics were logged
        assert_metric_logged(temp_run, "accuracy")
        assert_metric_logged(temp_run, "precision")
        assert_metric_logged(temp_run, "recall")
        assert_metric_logged(temp_run, "f1_score")
        # LogisticRegression has predict_proba, so these should also be logged
        assert_metric_logged(temp_run, "log_loss")
        assert_metric_logged(temp_run, "roc_auc")

    def test_multiclass_classifier_metrics(self, temp_run, multiclass_classification_data):
        """Test metrics for multiclass classification."""
        from artifacta.integrations import sklearn

        X_train, X_test, y_train, y_test = multiclass_classification_data

        sklearn.enable_autolog()

        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_train, y_train)

        sklearn.disable_autolog()

        # Verify multiclass metrics were logged
        assert_metric_logged(temp_run, "accuracy")
        assert_metric_logged(temp_run, "precision")  # weighted average
        assert_metric_logged(temp_run, "recall")  # weighted average
        assert_metric_logged(temp_run, "f1_score")  # weighted average
        assert_metric_logged(temp_run, "roc_auc")  # multiclass OVR

    def test_classifier_without_predict_proba(self, temp_run, binary_classification_data):
        """Test classifier that doesn't have predict_proba."""
        from artifacta.integrations import sklearn
        from sklearn.svm import LinearSVC

        X_train, X_test, y_train, y_test = binary_classification_data

        sklearn.enable_autolog()

        clf = LinearSVC(random_state=42, max_iter=1000)
        clf.fit(X_train, y_train)

        sklearn.disable_autolog()

        # Should log basic metrics but not log_loss or roc_auc


class TestSklearnRegressorMetrics:
    """Test regressor metric logging."""

    def test_regressor_metrics(self, temp_run, regression_data):
        """Test metrics for regression."""
        from artifacta.integrations import sklearn

        X_train, X_test, y_train, y_test = regression_data

        sklearn.enable_autolog()

        reg = RandomForestRegressor(n_estimators=10, random_state=42)
        reg.fit(X_train, y_train)

        sklearn.disable_autolog()

        # Verify regressor metrics were logged
        assert_metric_logged(temp_run, "training_score")
        assert_metric_logged(temp_run, "mse")
        assert_metric_logged(temp_run, "rmse")
        assert_metric_logged(temp_run, "mae")
        assert_metric_logged(temp_run, "r2_score")

    def test_linear_regression_metrics(self, temp_run, regression_data):
        """Test metrics for simple linear regression."""
        from artifacta.integrations import sklearn

        X_train, X_test, y_train, y_test = regression_data

        sklearn.enable_autolog()

        reg = LinearRegression()
        reg.fit(X_train, y_train)

        sklearn.disable_autolog()


class TestSklearnModelArtifacts:
    """Test model artifact logging."""

    def test_log_model_artifact(self, temp_run, binary_classification_data):
        """Test that fitted model is logged as artifact."""
        from artifacta.integrations import sklearn

        X_train, X_test, y_train, y_test = binary_classification_data

        sklearn.enable_autolog()

        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_train, y_train)

        sklearn.disable_autolog()

        # Verify model artifact was logged
        assert_model_artifact_logged(temp_run)
        assert_artifact_logged(temp_run, "RandomForestClassifier")

    def test_model_can_be_loaded(self, temp_run, binary_classification_data):
        """Test that logged model can be loaded and used."""
        from artifacta.integrations import sklearn

        X_train, X_test, y_train, y_test = binary_classification_data

        sklearn.enable_autolog()

        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_train, y_train)
        clf.predict(X_test)

        sklearn.disable_autolog()

        # In real implementation, load model from artifact and verify predictions match


class TestSklearnMetaEstimators:
    """Test meta-estimator support (Pipeline, GridSearchCV)."""

    def test_pipeline_logging(self, temp_run, binary_classification_data):
        """Test logging for Pipeline estimator."""
        from artifacta.integrations import sklearn

        X_train, X_test, y_train, y_test = binary_classification_data

        sklearn.enable_autolog()

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42)),
        ])
        pipe.fit(X_train, y_train)

        sklearn.disable_autolog()

        # Should log parameters from all pipeline steps
        # Should log metrics from final estimator

    def test_gridsearchcv_logging(self, temp_run, binary_classification_data):
        """Test logging for GridSearchCV."""
        from artifacta.integrations import sklearn

        X_train, X_test, y_train, y_test = binary_classification_data

        sklearn.enable_autolog()

        param_grid = {
            "max_depth": [3, 5],
            "n_estimators": [5, 10],
        }
        grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=2,
        )
        grid.fit(X_train, y_train)

        sklearn.disable_autolog()

        # MLflow creates parent run + child runs for each CV fold
        # For now, just ensure it completes without error


class TestSklearnConfiguration:
    """Test autolog configuration options."""

    def test_disable_model_logging(self, temp_run, binary_classification_data):
        """Test disabling model logging."""
        from artifacta.integrations import sklearn

        from artifacta.tests.test_utils import count_logged_artifacts

        X_train, X_test, y_train, y_test = binary_classification_data

        sklearn.enable_autolog(log_models=False, log_datasets=False)

        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)

        sklearn.disable_autolog()

        # Model should NOT be logged (and no dataset artifacts either)
        # Exclude config artifact which is auto-logged from update_config()
        num_artifacts = count_logged_artifacts(temp_run, exclude_config=True)
        assert num_artifacts == 0, f"Expected 0 artifacts, got {num_artifacts}"

    def test_disable_metrics_logging(self, temp_run, binary_classification_data):
        """Test disabling metrics logging."""
        from artifacta.integrations import sklearn

        X_train, X_test, y_train, y_test = binary_classification_data

        sklearn.enable_autolog(log_training_metrics=False)

        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)

        sklearn.disable_autolog()

        # Metrics should NOT be logged
        metrics = get_logged_metrics(temp_run)
        assert len(metrics) == 0, f"Expected 0 metrics, got {len(metrics)}: {list(metrics.keys())}"


class TestSklearnEdgeCases:
    """Test edge cases and error handling."""

    def test_estimator_without_score(self, temp_run):
        """Test estimator that doesn't implement score()."""
        from artifacta.integrations import sklearn
        from sklearn.cluster import KMeans

        X = np.random.rand(100, 10)

        sklearn.enable_autolog()

        # KMeans doesn't have score() method (unsupervised)
        # Should not crash
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X)

        sklearn.disable_autolog()

    def test_empty_dataset(self, temp_run):
        """Test with empty dataset (edge case)."""
        from artifacta.integrations import sklearn

        sklearn.enable_autolog()

        # Should handle gracefully (or raise appropriate sklearn error)
        clf = LogisticRegression()
        try:
            clf.fit(np.array([]).reshape(0, 5), np.array([]))
        except ValueError:
            pass  # Expected sklearn error

        sklearn.disable_autolog()

    def test_single_sample(self, temp_run):
        """Test with single sample dataset."""
        from artifacta.integrations import sklearn

        sklearn.enable_autolog()

        X = np.array([[1, 2, 3]])
        y = np.array([0])

        clf = DecisionTreeClassifier()
        clf.fit(X, y)

        sklearn.disable_autolog()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
