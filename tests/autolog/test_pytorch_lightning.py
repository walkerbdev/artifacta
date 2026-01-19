"""PyTorch Lightning autolog integration tests"""

import pytest

# Skip all tests if pytorch_lightning not available
try:
    import pytorch_lightning  # noqa: F401

    skip_pytorch = False
except ImportError:
    skip_pytorch = True

pytestmark = pytest.mark.skipif(skip_pytorch, reason="PyTorch Lightning not installed")


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
    assert "ArtifactaCheckpointCallback" in callback_names, (
        f"ArtifactaCheckpointCallback not found in {callback_names}"
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
    assert "ArtifactaCheckpointCallback" in callback_names1

    # Disable
    ds.disable_autolog()

    # Create trainer - should NOT have callback
    trainer2 = pl.Trainer(max_epochs=1, logger=False, enable_checkpointing=False)
    callback_names2 = [type(cb).__name__ for cb in trainer2.callbacks]
    assert "ArtifactaCheckpointCallback" not in callback_names2


def test_checkpoint_config_options():
    """Test checkpoint configuration options"""
    import pytorch_lightning as pl

    import artifacta as ds

    # Test with custom config
    ds.autolog(framework="pytorch")

    trainer = pl.Trainer(max_epochs=1, logger=False, enable_checkpointing=False)

    # Find Artifacta callback
    ds_callback = None
    for cb in trainer.callbacks:
        if type(cb).__name__ == "ArtifactaCheckpointCallback":
            ds_callback = cb
            break

    assert ds_callback is not None
    # Config options no longer passed to autolog - just verify callback exists

    # Cleanup
    ds.disable_autolog()
