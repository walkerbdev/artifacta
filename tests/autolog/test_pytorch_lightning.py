"""PyTorch Lightning autolog integration tests"""

import pytest

# Skip all tests if pytorch_lightning not available
try:
    import pytorch_lightning  # noqa: F401

    skip_pytorch = False
except ImportError:
    skip_pytorch = True

pytestmark = pytest.mark.skipif(skip_pytorch, reason="PyTorch Lightning not installed")


@pytest.fixture
def temp_run(monkeypatch):
    """Create and cleanup temporary run with mocked HTTP emitter."""
    from artifacta import init
    from artifacta.tests.test_utils import MockHTTPEmitter

    # Temporarily disable strict mode so init doesn't fail without server
    monkeypatch.delenv("ARTIFACTA_STRICT_MODE", raising=False)

    run = init(project="test_pytorch_lightning_autolog", name="test_run")
    # Replace the http_emitter with our mock
    run.http_emitter = MockHTTPEmitter(run.id)
    yield run
    run.finish()


def test_pytorch_lightning_callback_injection():
    """Test that autolog injects callback into Trainer"""
    import pytorch_lightning as pl
    import torch
    from torch import nn

    import artifacta as ds

    # Enable autolog
    ds.autolog(framework="pytorch")

    # Create simple model
    class DummyModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(10, 1)

        def forward(self, x):
            return self.layer(x)

        def training_step(self, batch, _):
            x, y = batch
            y_hat = self(x)
            loss = nn.functional.mse_loss(y_hat, y)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.001)

    # Create trainer
    trainer = pl.Trainer(max_epochs=1, logger=False, enable_checkpointing=False)

    # Check that callback was injected
    callback_names = [type(cb).__name__ for cb in trainer.callbacks]
    assert "ArtifactaAutologCallback" in callback_names, (
        f"ArtifactaAutologCallback not found in {callback_names}"
    )

    # Cleanup
    ds.disable_autolog()


def test_checkpoint_logging_requires_run():
    """Test that checkpoint logging only happens when run is active"""
    import pytorch_lightning as pl
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    import artifacta as ds

    # Enable autolog
    ds.autolog(framework="pytorch")

    # Create simple model
    class DummyModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(10, 1)

        def forward(self, x):
            return self.layer(x)

        def training_step(self, batch, _):
            x, y = batch
            y_hat = self(x)
            loss = nn.functional.mse_loss(y_hat, y)
            self.log("train_loss", loss)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.001)

    # Create dummy data
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=10)

    # Create trainer without run - should not crash
    model = DummyModel()
    trainer = pl.Trainer(max_epochs=1, logger=False, enable_checkpointing=False)
    trainer.fit(model, dataloader)

    # Cleanup
    ds.disable_autolog()


def test_enable_disable_autolog():
    """Test enabling and disabling autolog"""
    import pytorch_lightning as pl

    import artifacta as ds

    # Enable
    ds.autolog(framework="pytorch")

    # Create trainer - should have callback
    trainer1 = pl.Trainer(max_epochs=1, logger=False, enable_checkpointing=False)
    callback_names1 = [type(cb).__name__ for cb in trainer1.callbacks]
    assert "ArtifactaAutologCallback" in callback_names1

    # Disable
    ds.disable_autolog()

    # Create trainer - should NOT have callback
    trainer2 = pl.Trainer(max_epochs=1, logger=False, enable_checkpointing=False)
    callback_names2 = [type(cb).__name__ for cb in trainer2.callbacks]
    assert "ArtifactaAutologCallback" not in callback_names2


def test_parameter_logging(temp_run):
    """Test that autolog logs parameters (epochs, optimizer config)"""
    import pytorch_lightning as pl
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    import artifacta as ds

    # Enable autolog
    ds.autolog(framework="pytorch")

    # Create simple model
    class DummyModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(10, 1)

        def forward(self, x):
            return self.layer(x)

        def training_step(self, batch, _):
            x, y = batch
            y_hat = self(x)
            loss = nn.functional.mse_loss(y_hat, y)
            self.log("train_loss", loss)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.002, weight_decay=1e-5)

    # Create dummy data
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=10)

    # Train
    model = DummyModel()
    trainer = pl.Trainer(max_epochs=3, logger=False, enable_checkpointing=False)
    trainer.fit(model, dataloader)

    # Verify parameters were added to config
    config = temp_run.config
    assert config["epochs"] == 3
    assert config["optimizer_name"] == "Adam"
    assert config["lr"] == 0.002
    assert config["weight_decay"] == 1e-5

    # Cleanup
    ds.disable_autolog()


def test_metric_logging(temp_run):
    """Test that autolog logs metrics per epoch"""
    import pytorch_lightning as pl
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    import artifacta as ds

    # Enable autolog
    ds.autolog(framework="pytorch")

    # Create simple model that logs metrics
    class DummyModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(10, 1)

        def forward(self, x):
            return self.layer(x)

        def training_step(self, batch, _):
            x, y = batch
            y_hat = self(x)
            loss = nn.functional.mse_loss(y_hat, y)
            self.log("train_loss", loss)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.001)

    # Create dummy data
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=10)

    # Train
    model = DummyModel()
    trainer = pl.Trainer(max_epochs=3, logger=False, enable_checkpointing=False)
    trainer.fit(model, dataloader)

    # Verify metrics were logged as Series data
    # Check emitted_data for "training_metrics" series
    logged_data = False
    if hasattr(temp_run.http_emitter, 'emitted_data'):
        for event_type, data in temp_run.http_emitter.emitted_data:
            if event_type == "structured_data" and data.get("name") == "training_metrics":
                logged_data = True
                # Verify it has train_loss field in the fields dict
                series_data = data.get("data", {})
                fields = series_data.get("fields", series_data)
                assert "train_loss" in fields, "Should have train_loss in metrics"
                break
    assert logged_data, "Should have logged training_metrics as Series"

    # Cleanup
    ds.disable_autolog()


def test_final_model_logging(temp_run):
    """Test that autolog logs final trained model"""
    import pytorch_lightning as pl
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    import artifacta as ds

    # Enable autolog with model logging
    ds.autolog(framework="pytorch")

    # Create simple model
    class DummyModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(10, 1)

        def forward(self, x):
            return self.layer(x)

        def training_step(self, batch, _):
            x, y = batch
            y_hat = self(x)
            loss = nn.functional.mse_loss(y_hat, y)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.001)

    # Create dummy data
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=10)

    # Train
    model = DummyModel()
    trainer = pl.Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
    trainer.fit(model, dataloader)

    # Verify final model was logged
    artifacts = temp_run.http_emitter.emitted_artifacts
    model_artifacts = [a for a in artifacts if a.get("name") == "model"]
    assert len(model_artifacts) == 1, "Should have logged final model"
    assert model_artifacts[0]["metadata"]["artifact_type"] == "model"
    assert model_artifacts[0]["metadata"]["framework"] == "pytorch_lightning"

    # Cleanup
    ds.disable_autolog()


def test_disable_checkpoints(temp_run):
    """Test disabling checkpoint logging"""
    import pytorch_lightning as pl
    import torch
    from artifacta.integrations import pytorch_lightning as pl_integration
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    import artifacta as ds

    # Enable autolog with checkpoints disabled
    pl_integration.enable_autolog(log_checkpoints=False, log_models=True)

    # Create simple model
    class DummyModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(10, 1)

        def forward(self, x):
            return self.layer(x)

        def training_step(self, batch, _):
            x, y = batch
            y_hat = self(x)
            loss = nn.functional.mse_loss(y_hat, y)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.001)

    # Create dummy data
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=10)

    # Train
    model = DummyModel()
    trainer = pl.Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
    trainer.fit(model, dataloader)

    # Verify no checkpoints logged
    artifacts = temp_run.http_emitter.emitted_artifacts
    checkpoint_artifacts = [a for a in artifacts if "checkpoint" in a.get("name", "")]
    assert len(checkpoint_artifacts) == 0, "Should not log checkpoints when disabled"

    # But final model should still be logged
    model_artifacts = [a for a in artifacts if a.get("name") == "model"]
    assert len(model_artifacts) == 1, "Should still log final model"

    # Cleanup
    ds.disable_autolog()
