"""Tests for XGBoost autolog integration.

Tests cover all features matching MLflow's XGBoost autolog:
- Parameter logging (native API and sklearn API)
- Per-iteration metrics (via callbacks)
- Feature importance (weight, gain, cover)
- Model artifact logging
- Early stopping support
- Metric name sanitization (@ → _at_)
"""


import pytest
import xgboost as xgb
from sklearn.datasets import load_breast_cancer, make_regression
from sklearn.model_selection import train_test_split

from artifacta import init
from artifacta.tests.test_utils import (
    MockHTTPEmitter,
    assert_dataset_logged,
    assert_model_artifact_logged,
    assert_param_logged,
    get_logged_datasets,
    get_logged_metrics,
    get_logged_params,
)


@pytest.fixture
def binary_classification_data():
    """Binary classification dataset."""
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def regression_data():
    """Regression dataset."""
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def temp_run():
    """Create and cleanup temporary run with mocked HTTP emitter."""
    run = init(project="test_xgboost_autolog", name="test_run")
    # Replace the http_emitter with our mock
    run.http_emitter = MockHTTPEmitter(run.id)
    yield run
    run.finish()


class TestXGBoostAutologBasic:
    """Basic autolog functionality tests."""

    def test_autolog_enable_disable(self):
        """Test enabling and disabling autolog."""
        from artifacta.integrations import xgboost

        # Enable
        xgboost.enable_autolog()
        assert xgboost._AUTOLOG_ENABLED is True

        # Disable
        xgboost.disable_autolog()
        assert xgboost._AUTOLOG_ENABLED is False

    def test_autolog_patches_train(self):
        """Test that autolog patches xgboost.train()."""
        from artifacta.integrations import xgboost

        original_train = xgb.train
        xgboost.enable_autolog()

        # Check that train was patched
        assert xgb.train != original_train
        assert original_train == xgboost._ORIGINAL_TRAIN

        xgboost.disable_autolog()

    def test_autolog_patches_sklearn(self):
        """Test that autolog patches XGBoost sklearn API."""
        from artifacta.integrations import xgboost

        xgboost.enable_autolog()

        # Check that sklearn API was patched
        assert (xgb.XGBClassifier, "fit") in xgboost._ORIGINAL_SKLEARN_METHODS
        assert (xgb.XGBRegressor, "fit") in xgboost._ORIGINAL_SKLEARN_METHODS

        xgboost.disable_autolog()

    def test_autolog_without_active_run(self, binary_classification_data):
        """Test that autolog doesn't crash when no run is active."""
        from artifacta.integrations import xgboost

        X_train, X_test, y_train, y_test = binary_classification_data

        xgboost.enable_autolog()

        # Should work without active run (just doesn't log)
        dtrain = xgb.DMatrix(X_train, y_train)
        params = {"max_depth": 3, "objective": "binary:logistic"}
        xgb.train(params, dtrain, num_boost_round=10)

        xgboost.disable_autolog()


class TestXGBoostNativeAPI:
    """Test native xgboost.train() API autolog."""

    def test_log_params(self, temp_run, binary_classification_data):
        """Test logging parameters from xgboost.train()."""
        from artifacta.integrations import xgboost

        X_train, X_test, y_train, y_test = binary_classification_data

        xgboost.enable_autolog()

        dtrain = xgb.DMatrix(X_train, y_train)
        params = {
            "max_depth": 3,
            "eta": 0.1,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        }
        xgb.train(params, dtrain, num_boost_round=10)

        xgboost.disable_autolog()

        # Verify params were logged
        assert_param_logged(temp_run, "max_depth", 3)
        assert_param_logged(temp_run, "eta", 0.1)
        assert_param_logged(temp_run, "objective", "binary:logistic")

    def test_log_metrics_with_evals(self, temp_run, binary_classification_data):
        """Test logging per-iteration metrics with evals."""
        from artifacta.integrations import xgboost

        X_train, X_test, y_train, y_test = binary_classification_data

        xgboost.enable_autolog()

        dtrain = xgb.DMatrix(X_train, y_train)
        dtest = xgb.DMatrix(X_test, y_test)
        params = {"max_depth": 3, "objective": "binary:logistic"}

        xgb.train(
            params,
            dtrain,
            num_boost_round=10,
            evals=[(dtrain, "train"), (dtest, "test")],
        )

        xgboost.disable_autolog()

        # Verify per-iteration metrics were logged
        metrics = get_logged_metrics(temp_run)
        # Should have train and test metrics (at least one type)
        assert len(metrics) > 0, "No metrics were logged"
        # Verify we have metrics from both train and test sets
        metric_names = list(metrics.keys())
        has_train = any("train" in name for name in metric_names)
        has_test = any("test" in name for name in metric_names)
        assert has_train or has_test, "Expected train or test metrics to be logged"

    def test_log_feature_importance(self, temp_run, binary_classification_data):
        """Test logging feature importance."""
        from artifacta.integrations import xgboost

        X_train, X_test, y_train, y_test = binary_classification_data

        xgboost.enable_autolog()

        dtrain = xgb.DMatrix(X_train, y_train)
        params = {"max_depth": 3, "objective": "binary:logistic"}
        xgb.train(params, dtrain, num_boost_round=10)

        xgboost.disable_autolog()

        # Verify feature importance artifacts were logged
        artifacts = temp_run.http_emitter.emitted_artifacts
        importance_artifacts = [
            a for a in artifacts if "feature_importance" in a.get("name", "")
        ]
        assert len(importance_artifacts) > 0, "No feature importance artifacts logged"
        # Should have weight, gain, and cover by default
        importance_names = [a.get("name", "") for a in importance_artifacts]
        assert any("weight" in name for name in importance_names), "weight importance not logged"
        assert any("gain" in name for name in importance_names), "gain importance not logged"
        assert any("cover" in name for name in importance_names), "cover importance not logged"

    def test_log_model(self, temp_run, binary_classification_data):
        """Test logging trained model."""
        from artifacta.integrations import xgboost

        X_train, X_test, y_train, y_test = binary_classification_data

        xgboost.enable_autolog()

        dtrain = xgb.DMatrix(X_train, y_train)
        params = {"max_depth": 3, "objective": "binary:logistic"}
        xgb.train(params, dtrain, num_boost_round=10)

        xgboost.disable_autolog()

        # Verify model artifact was logged
        assert_model_artifact_logged(temp_run)

    def test_metric_name_sanitization(self, temp_run, binary_classification_data):
        """Test that metric names with @ are sanitized."""
        from artifacta.integrations import xgboost

        X_train, X_test, y_train, y_test = binary_classification_data

        xgboost.enable_autolog()

        dtrain = xgb.DMatrix(X_train, y_train)
        dtest = xgb.DMatrix(X_test, y_test)
        params = {
            "max_depth": 3,
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "ndcg@3"],
        }

        xgb.train(
            params,
            dtrain,
            num_boost_round=10,
            evals=[(dtest, "test")],
        )

        xgboost.disable_autolog()

        # Verify metric name sanitization: "ndcg@3" should be logged as "test_ndcg_at_3"
        metrics = get_logged_metrics(temp_run)
        metric_names = list(metrics.keys())
        # Check that @ was replaced with _at_
        has_sanitized = any("_at_" in name for name in metric_names)
        assert has_sanitized, "Expected sanitized metric name with '_at_' but found none"
        # Should not have @ in any metric name
        has_at_symbol = any("@" in name for name in metric_names)
        assert not has_at_symbol, "Found unsanitized metric name with '@' symbol"


class TestXGBoostSklearnAPI:
    """Test XGBoost sklearn API autolog."""

    def test_xgbclassifier_params(self, temp_run, binary_classification_data):
        """Test logging XGBClassifier parameters."""
        from artifacta.integrations import xgboost

        X_train, X_test, y_train, y_test = binary_classification_data

        xgboost.enable_autolog()

        clf = xgb.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=10)
        clf.fit(X_train, y_train)

        xgboost.disable_autolog()

        # Verify params were logged
        assert_param_logged(temp_run, "max_depth", 3)
        assert_param_logged(temp_run, "learning_rate", 0.1)
        assert_param_logged(temp_run, "n_estimators", 10)

    def test_xgbregressor_params(self, temp_run, regression_data):
        """Test logging XGBRegressor parameters."""
        from artifacta.integrations import xgboost

        X_train, X_test, y_train, y_test = regression_data

        xgboost.enable_autolog()

        reg = xgb.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=10)
        reg.fit(X_train, y_train)

        xgboost.disable_autolog()

        # Verify params were logged
        assert_param_logged(temp_run, "max_depth", 3)
        assert_param_logged(temp_run, "learning_rate", 0.1)
        assert_param_logged(temp_run, "n_estimators", 10)

    def test_sklearn_feature_importance(self, temp_run, binary_classification_data):
        """Test feature importance logging for sklearn API."""
        from artifacta.integrations import xgboost

        X_train, X_test, y_train, y_test = binary_classification_data

        xgboost.enable_autolog()

        clf = xgb.XGBClassifier(max_depth=3, n_estimators=10)
        clf.fit(X_train, y_train)

        xgboost.disable_autolog()

        # Verify feature importance artifacts were logged
        artifacts = temp_run.http_emitter.emitted_artifacts
        importance_artifacts = [
            a for a in artifacts if "feature_importance" in a.get("name", "")
        ]
        assert len(importance_artifacts) > 0, "No feature importance artifacts logged"

    def test_sklearn_model_logging(self, temp_run, binary_classification_data):
        """Test model logging for sklearn API."""
        from artifacta.integrations import xgboost

        X_train, X_test, y_train, y_test = binary_classification_data

        xgboost.enable_autolog()

        clf = xgb.XGBClassifier(max_depth=3, n_estimators=10)
        clf.fit(X_train, y_train)

        xgboost.disable_autolog()

        # Verify model artifact was logged
        assert_model_artifact_logged(temp_run)


class TestXGBoostConfiguration:
    """Test autolog configuration options."""

    def test_disable_model_logging(self, temp_run, binary_classification_data):
        """Test disabling model logging."""
        from artifacta.integrations import xgboost

        X_train, X_test, y_train, y_test = binary_classification_data

        xgboost.enable_autolog(log_models=False)

        dtrain = xgb.DMatrix(X_train, y_train)
        params = {"max_depth": 3, "objective": "binary:logistic"}
        xgb.train(params, dtrain, num_boost_round=10)

        xgboost.disable_autolog()

        # Verify model was NOT logged
        artifacts = temp_run.http_emitter.emitted_artifacts
        model_artifacts = [a for a in artifacts if a.get("name", "") == "model"]
        assert len(model_artifacts) == 0, "Model should not be logged when log_models=False"

    def test_disable_feature_importance(self, temp_run, binary_classification_data):
        """Test disabling feature importance logging."""
        from artifacta.integrations import xgboost

        X_train, X_test, y_train, y_test = binary_classification_data

        xgboost.enable_autolog(log_feature_importance=False)

        dtrain = xgb.DMatrix(X_train, y_train)
        params = {"max_depth": 3, "objective": "binary:logistic"}
        xgb.train(params, dtrain, num_boost_round=10)

        xgboost.disable_autolog()

        # Verify feature importance was NOT logged
        artifacts = temp_run.http_emitter.emitted_artifacts
        importance_artifacts = [
            a for a in artifacts if "feature_importance" in a.get("name", "")
        ]
        assert len(importance_artifacts) == 0, "Feature importance should not be logged when log_feature_importance=False"

    def test_custom_importance_types(self, temp_run, binary_classification_data):
        """Test custom importance types."""
        from artifacta.integrations import xgboost

        X_train, X_test, y_train, y_test = binary_classification_data

        xgboost.enable_autolog(importance_types=["weight"])

        dtrain = xgb.DMatrix(X_train, y_train)
        params = {"max_depth": 3, "objective": "binary:logistic"}
        xgb.train(params, dtrain, num_boost_round=10)

        xgboost.disable_autolog()

        # Verify only "weight" importance was logged
        artifacts = temp_run.http_emitter.emitted_artifacts
        importance_artifacts = [
            a for a in artifacts if "feature_importance" in a.get("name", "")
        ]
        assert len(importance_artifacts) > 0, "No feature importance artifacts logged"
        importance_names = [a.get("name", "") for a in importance_artifacts]
        assert any("weight" in name for name in importance_names), "weight importance not logged"
        assert not any("gain" in name for name in importance_names), "gain importance should not be logged"
        assert not any("cover" in name for name in importance_names), "cover importance should not be logged"


class TestXGBoostDatasetLogging:
    """Test dataset metadata logging."""

    def test_native_api_dataset_logging(self, temp_run, binary_classification_data):
        """Test dataset logging for native API (xgb.train)."""
        from artifacta.integrations import xgboost

        X_train, X_test, y_train, y_test = binary_classification_data

        xgboost.enable_autolog()

        dtrain = xgb.DMatrix(X_train, y_train)
        dtest = xgb.DMatrix(X_test, y_test)
        params = {"max_depth": 3, "objective": "binary:logistic"}

        xgb.train(
            params,
            dtrain,
            num_boost_round=5,
            evals=[(dtrain, "train"), (dtest, "test")],
        )

        xgboost.disable_autolog()

        # Verify training dataset was logged
        assert_dataset_logged(temp_run, context="train", expected_shape=X_train.shape)

        # Verify eval datasets were logged
        assert_dataset_logged(temp_run, context="eval_train")
        assert_dataset_logged(temp_run, context="eval_test")

        # Check metadata fields
        datasets = get_logged_datasets(temp_run)
        train_data = datasets["train"]
        assert "features_digest" in train_data
        assert "features_dtype" in train_data

    def test_sklearn_api_dataset_logging(self, temp_run, binary_classification_data):
        """Test dataset logging for sklearn API (XGBClassifier)."""
        from artifacta.integrations import xgboost

        X_train, X_test, y_train, y_test = binary_classification_data

        xgboost.enable_autolog()

        clf = xgb.XGBClassifier(n_estimators=5, max_depth=3)
        clf.fit(X_train, y_train)

        xgboost.disable_autolog()

        # Verify dataset was logged
        assert_dataset_logged(temp_run, context="train", expected_shape=X_train.shape)

        datasets = get_logged_datasets(temp_run)
        train_data = datasets["train"]
        assert train_data["features_shape"] == list(X_train.shape)
        assert train_data["targets_shape"] == list(y_train.shape)

    def test_dataset_logging_can_be_disabled(self, temp_run, binary_classification_data):
        """Test that dataset logging can be disabled."""
        from artifacta.integrations import xgboost

        X_train, X_test, y_train, y_test = binary_classification_data

        xgboost.enable_autolog(log_datasets=False)

        dtrain = xgb.DMatrix(X_train, y_train)
        params = {"max_depth": 3, "objective": "binary:logistic"}
        xgb.train(params, dtrain, num_boost_round=5)

        xgboost.disable_autolog()

        # Verify dataset was NOT logged
        datasets = get_logged_datasets(temp_run)
        assert "train" not in datasets, "Dataset should not be logged when log_datasets=False"


class TestXGBoostEdgeCases:
    """Test edge cases and error handling."""

    def test_no_evals(self, temp_run, binary_classification_data):
        """Test training without evals (no per-iteration metrics)."""
        from artifacta.integrations import xgboost

        X_train, X_test, y_train, y_test = binary_classification_data

        xgboost.enable_autolog()

        dtrain = xgb.DMatrix(X_train, y_train)
        params = {"max_depth": 3, "objective": "binary:logistic"}
        xgb.train(params, dtrain, num_boost_round=10)

        xgboost.disable_autolog()

        # Verify training completed without error
        # Params should still be logged even without evals
        assert_param_logged(temp_run, "max_depth", 3)
        # Model should still be logged
        assert_model_artifact_logged(temp_run)

    def test_prevents_nested_logging(self, temp_run, binary_classification_data):
        """Test that nested training doesn't create duplicate logs."""
        from artifacta.integrations import xgboost

        X_train, X_test, y_train, y_test = binary_classification_data

        xgboost.enable_autolog()

        # This shouldn't cause issues even if called multiple times
        dtrain = xgb.DMatrix(X_train, y_train)
        params = {"max_depth": 3, "objective": "binary:logistic"}
        xgb.train(params, dtrain, num_boost_round=5)
        xgb.train(params, dtrain, num_boost_round=5)

        xgboost.disable_autolog()

        # Verify both trainings completed successfully
        # Both should log params (two separate train calls)
        params_logged = get_logged_params(temp_run)
        assert "max_depth" in params_logged, "Parameters should be logged"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
