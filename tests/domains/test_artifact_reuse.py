"""
Test artifact reuse across runs - Output from one run becomes input to another
Demonstrates lineage tracking: Train → Model → Inference

Run with: pytest tests/domains/test_artifact_reuse.py -v
"""

import os
import tempfile
import time

import pytest

import artifacta as ds


@pytest.mark.e2e
def test_artifact_chain():
    """Test that an artifact output from one run can be input to another run"""

    print("\n=== Artifact Reuse Chain Test ===\n")

    # Create a temporary model file that will be passed between runs
    model_dir = tempfile.mkdtemp()
    model_path = os.path.join(model_dir, "trained_model.pt")

    with open(model_path, "w") as f:
        f.write("MODEL CHECKPOINT\n")
        f.write("Epoch: 100\n")
        f.write("Loss: 0.001\n")
        f.write("Accuracy: 0.95\n")

    try:
        # ============================================================
        # RUN 1: TRAINING - Produces model as OUTPUT
        # ============================================================
        print("Run 1: Training")
        run_train = ds.init(
            project="ml-pipeline",
            name="train-resnet",
            config={"epochs": 100, "lr": 0.001, "batch_size": 32},
        )
        print(f"   Created: {run_train.id}")

        # Log training metrics
        ds.log(
            "accuracy",
            ds.Series(
                index="epoch",
                fields={"train": [0.7, 0.85, 0.92, 0.95], "val": [0.65, 0.80, 0.88, 0.93]},
                index_values=[25, 50, 75, 100],
            ),
        )

        # Log the trained model as OUTPUT
        run_train.log_output(model_path, name="trained_model")
        print("   Logged: trained_model.pt as OUTPUT")

        print("   Training complete\n")
        time.sleep(0.5)

        # ============================================================
        # RUN 2: INFERENCE - Uses model as INPUT
        # ============================================================
        print("Run 2: Inference")
        run_inference = ds.init(
            project="ml-pipeline",
            name="inference-batch-1",
            config={"batch_size": 64, "model_checkpoint": "trained_model.pt"},
        )
        print(f"   Created: {run_inference.id}")

        # Log the SAME model file as INPUT (should reuse artifact by hash!)
        run_inference.log_input(model_path, name="trained_model")
        print("   Logged: trained_model.pt as INPUT")

        # Log inference results
        ds.log(
            "predictions",
            ds.Table(
                columns=[
                    {"name": "image_id", "type": "string"},
                    {"name": "predicted_class", "type": "string"},
                    {"name": "confidence", "type": "number"},
                ],
                data=[
                    ["img_001", "cat", 0.95],
                    ["img_002", "dog", 0.88],
                    ["img_003", "bird", 0.92],
                ],
            ),
        )

        print("   Inference complete\n")
        time.sleep(0.5)

        # ============================================================
        # RUN 3: ANOTHER INFERENCE - Also uses same model as INPUT
        # ============================================================
        print("Run 3: Inference (Batch 2)")
        run_inference_2 = ds.init(
            project="ml-pipeline",
            name="inference-batch-2",
            config={"batch_size": 128, "model_checkpoint": "trained_model.pt"},
        )
        print(f"   Created: {run_inference_2.id}")

        # Log the SAME model file as INPUT again
        run_inference_2.log_input(model_path, name="trained_model")
        print("   Logged: trained_model.pt as INPUT")

        # Log inference results
        ds.log(
            "predictions",
            ds.Table(
                columns=[
                    {"name": "image_id", "type": "string"},
                    {"name": "predicted_class", "type": "string"},
                    {"name": "confidence", "type": "number"},
                ],
                data=[
                    ["img_010", "horse", 0.91],
                    ["img_011", "cat", 0.87],
                ],
            ),
        )

        print("   Inference complete\n")

        print("=" * 60)
        print("TEST COMPLETE!")
        print("=" * 60)
        print("\nCheck the UI Lineage view (select all 3 runs):")
        print("   Expected graph:")
        print()
        print("                     trained_model.pt")
        print("                            |")
        print("   train-resnet → inference-batch-1")
        print("                            → inference-batch-2")
        print()
        print("   - train-resnet has trained_model as OUTPUT (right side)")
        print("   - inference-batch-1 has trained_model as INPUT (left side)")
        print("   - inference-batch-2 has trained_model as INPUT (left side)")
        print("   - SAME artifact connects all 3 runs (reused by hash)")
        print()

    finally:
        # Cleanup
        if os.path.exists(model_path):
            os.unlink(model_path)
        if os.path.exists(model_dir):
            os.rmdir(model_dir)
