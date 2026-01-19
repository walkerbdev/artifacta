# artifacta

Experiment tracking package for Artifacta.

## Installation

```bash
pip install -e .
```

## Usage

```python
import artifacta as ds

# Initialize run
ds.init(project="my-project", config={"lr": 0.001})

# Log metrics during training
for epoch in range(100):
    loss = train_step()
    ds.log({"epoch": epoch, "loss": loss})

# Finish run
ds.finish()
```

## Features

- Automatic metadata capture (git, environment, system)
- Background system monitoring (CPU, GPU, memory)
- SQLite database for run history
- JSONL files for detailed metrics
- Works offline, no server required
