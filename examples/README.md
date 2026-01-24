# Artifacta Examples

This directory contains comprehensive examples demonstrating all Artifacta features and use cases.

## Installation

Install Artifacta with all dependencies needed to run the examples:

```bash
# Create a new virtual environment
python3 -m venv artifacta-examples
source artifacta-examples/bin/activate  # On Windows: artifacta-examples\Scripts\activate

# Install artifacta + ML frameworks (sklearn, xgboost, pytorch, tensorflow)
pip install -r examples/requirements.txt
```

## Quick Start

**Start the UI server** (in a separate terminal):
```bash
artifacta ui
# View at http://localhost:8000
```

**Run the minimal example**:
```bash
python examples/core/01_basic_tracking.py
```

**Run all examples**:
```bash
python examples/run_all_examples.py
```
