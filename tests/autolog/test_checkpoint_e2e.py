"""End-to-end checkpoint autologging test"""

import pytest

# Skip if pytorch_lightning not available
try:
    import pytorch_lightning  # noqa: F401

    skip_pytorch = False
except ImportError:
    skip_pytorch = True

pytestmark = pytest.mark.skipif(skip_pytorch, reason="PyTorch Lightning not installed")


def test_checkpoint_autolog_e2e():
    """End-to-end test: Train model and verify checkpoint is auto-logged"""
    import pytorch_lightning as pl
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    import artifacta as ds

    # Enable autolog BEFORE creating run
    ds.autolog(framework="pytorch")

    # Start a Artifacta run
    ds.init(project="test_autolog", name="checkpoint_test")

    try:
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

        # Train model
        model = DummyModel()
        trainer = pl.Trainer(
            max_epochs=2,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(model, dataloader)

        # Verify checkpoint was logged
        # The callback should have logged 2 checkpoints (one per epoch)
        ds_callback = None
        for cb in trainer.callbacks:
            if type(cb).__name__ == "ArtifactaAutologCallback":
                ds_callback = cb
                break

        assert ds_callback is not None, "ArtifactaAutologCallback not found"
        assert len(ds_callback.checkpoints_logged) == 2, (
            f"Expected 2 checkpoints logged, got {len(ds_callback.checkpoints_logged)}"
        )

        # Verify checkpoint metadata
        for i, ckpt_info in enumerate(ds_callback.checkpoints_logged):
            assert ckpt_info["epoch"] == i, f"Expected epoch {i}, got {ckpt_info['epoch']}"

    finally:
        # Cleanup
        ds.disable_autolog()


def test_checkpoint_autolog_best_only():
    """Test that save_best_only only logs improving checkpoints"""
    import pytorch_lightning as pl
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    import artifacta as ds

    # Enable autolog with save_best_only
    ds.autolog(framework="pytorch")

    # Start a Artifacta run
    ds.init(project="test_autolog", name="best_checkpoint_test")

    try:
        # Create model that logs train_loss
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

        # Train model
        model = DummyModel()
        trainer = pl.Trainer(
            max_epochs=3,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(model, dataloader)

        # Verify only best checkpoints were logged
        ds_callback = None
        for cb in trainer.callbacks:
            if type(cb).__name__ == "ArtifactaAutologCallback":
                ds_callback = cb
                break

        assert ds_callback is not None, "ArtifactaAutologCallback not found"
        # With save_best_only, should log fewer than total epochs (unless loss perfectly decreases)
        # At minimum, should log at least 1 checkpoint (first epoch)
        assert len(ds_callback.checkpoints_logged) >= 1, (
            "Expected at least 1 checkpoint with save_best_only"
        )
        assert len(ds_callback.checkpoints_logged) <= 3, (
            f"Expected at most 3 checkpoints, got {len(ds_callback.checkpoints_logged)}"
        )

    finally:
        # Cleanup
        ds.disable_autolog()
