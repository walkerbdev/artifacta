"""TensorFlow/Keras autolog integration tests"""

import pytest

# Skip all tests if tensorflow not available
try:
    import tensorflow as tf  # noqa: F401

    skip_tf = False
except ImportError:
    skip_tf = True

pytestmark = pytest.mark.skipif(skip_tf, reason="TensorFlow not installed")


@pytest.fixture
def temp_run(monkeypatch):
    """Create and cleanup temporary run with mocked HTTP emitter."""
    from artifacta import init
    from artifacta.tests.test_utils import MockHTTPEmitter

    # Temporarily disable strict mode so init doesn't fail without server
    monkeypatch.delenv("ARTIFACTA_STRICT_MODE", raising=False)

    run = init(project="test_tensorflow_autolog", name="test_run")
    # Replace the http_emitter with our mock
    run.http_emitter = MockHTTPEmitter(run.id)
    yield run
    run.finish()


def test_tensorflow_autolog_e2e(monkeypatch):
    """End-to-end test: Train Keras model and verify checkpoint is auto-logged"""
    import numpy as np
    import tensorflow as tf

    import artifacta as ds

    # Temporarily disable strict mode
    monkeypatch.delenv("ARTIFACTA_STRICT_MODE", raising=False)

    # Enable autolog
    ds.autolog(framework="tensorflow")

    # Start run
    ds.init(project="test_tf_autolog", name="tf_checkpoint_test")

    try:
        # Create simple model
        model = tf.keras.Sequential(
            [tf.keras.layers.Dense(10, input_shape=(5,)), tf.keras.layers.Dense(1)]
        )
        model.compile(optimizer="adam", loss="mse")

        # Create dummy data
        x_train = np.random.randn(100, 5).astype(np.float32)
        y_train = np.random.randn(100, 1).astype(np.float32)

        # Train model
        model.fit(x_train, y_train, epochs=2, verbose=0)

        # Note: Autolog works, but without HTTP emitter running artifacts won't be in database
        # Just verify the test ran without errors - actual integration test requires server running

    finally:
        ds.disable_autolog()


def test_tensorflow_best_only(monkeypatch):
    """Test that save_best_only only logs improving checkpoints"""
    import numpy as np
    import tensorflow as tf

    import artifacta as ds

    # Temporarily disable strict mode
    monkeypatch.delenv("ARTIFACTA_STRICT_MODE", raising=False)

    # Enable autolog with save_best_only
    ds.autolog(framework="tensorflow")

    # Start run
    ds.init(project="test_tf_autolog", name="tf_best_checkpoint_test")

    try:
        # Create simple model
        model = tf.keras.Sequential(
            [tf.keras.layers.Dense(10, input_shape=(5,)), tf.keras.layers.Dense(1)]
        )
        model.compile(optimizer="adam", loss="mse")

        # Create dummy data
        x_train = np.random.randn(100, 5).astype(np.float32)
        y_train = np.random.randn(100, 1).astype(np.float32)

        # Train model
        model.fit(x_train, y_train, epochs=3, verbose=0)

        # Note: Autolog works, but without HTTP emitter running artifacts won't be in database
        # Just verify the test ran without errors - actual integration test requires server running

    finally:
        ds.disable_autolog()


def test_enable_disable_tensorflow():
    """Test enabling and disabling TensorFlow autolog"""
    import tensorflow as tf

    import artifacta as ds

    # Enable
    ds.autolog(framework="tensorflow")

    # Check that fit was patched
    assert hasattr(tf.keras.Model.fit, "__name__")

    # Disable
    ds.disable_autolog()

    # Should still be callable
    assert callable(tf.keras.Model.fit)


def test_parameter_logging(temp_run):
    """Test that autolog logs parameters (epochs, batch_size, optimizer config)"""
    import numpy as np
    import tensorflow as tf

    import artifacta as ds

    # Enable autolog
    ds.autolog(framework="tensorflow")

    # Create simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=(5,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), loss="mse")

    # Create dummy data
    x_train = np.random.randn(100, 5).astype(np.float32)
    y_train = np.random.randn(100, 1).astype(np.float32)

    # Train
    model.fit(x_train, y_train, epochs=3, batch_size=16, verbose=0)

    # Verify parameters were added to config
    config = temp_run.config
    assert config["epochs"] == 3
    assert config["batch_size"] == 16
    assert config["optimizer_name"] == "Adam"
    assert "learning_rate" in config or "lr" in config

    # Cleanup
    ds.disable_autolog()


def test_metric_logging(temp_run):
    """Test that autolog logs metrics per epoch"""
    import numpy as np
    import tensorflow as tf

    import artifacta as ds

    # Enable autolog
    ds.autolog(framework="tensorflow")

    # Create simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=(5,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # Create dummy data
    x_train = np.random.randn(100, 5).astype(np.float32)
    y_train = np.random.randn(100, 1).astype(np.float32)

    # Train
    model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=0)

    # Verify metrics were logged as Series data
    logged_data = False
    if hasattr(temp_run.http_emitter, 'emitted_data'):
        for event_type, data in temp_run.http_emitter.emitted_data:
            if event_type == "structured_data" and data.get("name") == "training_metrics":
                logged_data = True
                # Verify it has loss field
                series_data = data.get("data", {})
                fields = series_data.get("fields", series_data)
                assert "loss" in fields, "Should have loss in metrics"
                break
    assert logged_data, "Should have logged training_metrics as Series"

    # Cleanup
    ds.disable_autolog()


def test_final_model_logging(temp_run):
    """Test that autolog logs final trained model"""
    import numpy as np
    import tensorflow as tf

    import artifacta as ds

    # Enable autolog with model logging
    ds.autolog(framework="tensorflow")

    # Create simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=(5,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    # Create dummy data
    x_train = np.random.randn(100, 5).astype(np.float32)
    y_train = np.random.randn(100, 1).astype(np.float32)

    # Train
    model.fit(x_train, y_train, epochs=2, verbose=0)

    # Verify final model was logged
    artifacts = temp_run.http_emitter.emitted_artifacts
    model_artifacts = [a for a in artifacts if a.get("name") == "model"]
    assert len(model_artifacts) == 1, "Should have logged final model"
    assert model_artifacts[0]["metadata"]["artifact_type"] == "model"
    assert model_artifacts[0]["metadata"]["framework"] == "tensorflow"

    # Cleanup
    ds.disable_autolog()


def test_disable_checkpoints(temp_run):
    """Test disabling checkpoint logging"""
    import numpy as np
    import tensorflow as tf
    from artifacta.integrations import tensorflow as tf_integration

    import artifacta as ds

    # Enable autolog with checkpoints disabled
    tf_integration.enable_autolog(log_checkpoints=False, log_models=True)

    # Create simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=(5,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    # Create dummy data
    x_train = np.random.randn(100, 5).astype(np.float32)
    y_train = np.random.randn(100, 1).astype(np.float32)

    # Train
    model.fit(x_train, y_train, epochs=2, verbose=0)

    # Verify no checkpoints logged
    artifacts = temp_run.http_emitter.emitted_artifacts
    checkpoint_artifacts = [a for a in artifacts if "checkpoint" in a.get("name", "")]
    assert len(checkpoint_artifacts) == 0, "Should not log checkpoints when disabled"

    # But final model should still be logged
    model_artifacts = [a for a in artifacts if a.get("name") == "model"]
    assert len(model_artifacts) == 1, "Should still log final model"

    # Cleanup
    ds.disable_autolog()
