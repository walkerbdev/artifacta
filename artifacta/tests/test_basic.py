"""Basic tests for artifacta package"""

import time

import artifacta as ds


def test_basic_usage():
    """Test basic init/log/finish flow"""

    # Initialize
    run = ds.init(project="test", config={"lr": 0.001, "epochs": 10})
    assert run is not None
    assert run.project == "test"

    # Log some metrics
    for i in range(10):
        ds.log({"epoch": i, "loss": 1.0 / (i + 1), "accuracy": i * 0.1})
        time.sleep(0.1)

    # Finish
    ds.finish()

    # Check files were created
    assert (run.log_dir / "metadata.json").exists()
    assert (run.log_dir / "metrics.jsonl").exists()
    assert (run.log_dir / "system.jsonl").exists()

    print(f"✅ Test passed! Files created at: {run.log_dir}")


if __name__ == "__main__":
    test_basic_usage()
