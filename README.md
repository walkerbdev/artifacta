<div align="center">

# Artifacta

**Universal experiment and artifact tracking — gain insights and optimize models with confidence.**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Elastic%202.0-green)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

---

## The Problem

Modern data science and machine learning workflows involve countless experiments—tweaking hyperparameters, adjusting data preprocessing, testing different architectures, updating dependencies, modifying code. **Every change produces different results**, but tracking and comparing these variations manually becomes overwhelming:

- Which parameters, environment, or code version led to that breakthrough result last week?
- How does changing the learning rate affect convergence across multiple runs?
- What's the actual performance difference between model architectures?
- Which preprocessing steps improved accuracy by 2%?
- Did upgrading that dependency break model performance?
- What code changes caused the regression?

Without systematic tracking of **parameters, metrics, code changes, dependencies, and environment**, you're flying blind—relying on scattered notes, terminal output, and memory.

But even with tracking, making sense of the data is hard:

- Manually comparing metrics across dozens of experiments
- Spotting patterns in hyperparameter sweeps
- Understanding why one approach outperformed another
- Deciding which direction to explore next

**Artifacta solves both problems** by automatically capturing experiments AND helping you understand them—with intelligent visualizations, multi-run comparisons, and LLM-powered analysis to explain what happened and why.

---

## Ecosystem & Alternatives

| Feature | Artifacta | MLflow | W&B | Neptune | Comet |
|---------|-----------|--------|-----|---------|-------|
| Zero-config install | ✅ | ❌ | ❌ | ❌ | ❌ |
| 100% offline | ✅ | ✅ | ❌ | ❌ | ❌ |
| Built-in lab notebook | ✅ | ❌ | ❌ | ❌ | ❌ |
| AI assistant (LiteLLM) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Domain-agnostic | ✅ | ✅ | ❌ | ⚠️ | ❌ |
| Framework autolog | ✅ | ✅ | ✅ | ✅ | ✅ |
| Team collaboration | ❌ | ⚠️ | ✅ | ✅ | ✅ |
| Model deployment | ❌ | ✅ | ✅ | ✅ | ✅ |

**Legend:** ✅ Yes | ⚠️ Partial | ❌ No

For detailed feature comparisons (deployment, visualization, integrations), see the [full documentation](https://docs.artifacta.ai).

---

## Why Choose Artifacta?

**What makes Artifacta different:**

- **Zero configuration**: Pre-built UI bundled with Python package—`pip install` and you're done. No Node.js, Docker, or build tools required
- **Truly offline-first**: Works 100% locally without any cloud dependencies, license servers, or internet connection
- **Server-side plot generation**: Log data primitives (Series, Scatter, Matrix), not matplotlib figures—Artifacta renders plots for you. No need to create and upload images (though you can if you want)
- **Built-in electronic lab notebook**: Rich text editor with LaTeX support, file attachments, and per-project organization—not available in any competitor
- **AI chat interface**: Built-in LLM chat (OpenAI, Anthropic, local models) to analyze experiments, results, and code. W&B and Comet have AI features in premium tiers only
- **Domain-agnostic design**: Primitives work for any field—ML, A/B tests, physics, finance, genomics, climate science. Not ML-only like most alternatives
- **Rich artifact previews**: Built-in viewers for video, audio, PDFs, code, images. MLflow only previews images; others require external viewers
- **Interactive artifact lineage**: Visual flow graph showing how artifacts relate. MLflow has no lineage visualization

---

## Visual Overview

**Automatic Plot Discovery**

![Plots](docs/_static/Plots.gif)

Artifacta automatically generates visualizations based on your data shape and metadata. No manual plot configuration needed.

**Artifact Management**

![Artifacts](docs/_static/Artifacts_1.gif)

Browse and preview datasets, models, code, images, videos, and documents with built-in file viewers.

---

## Quick Start

### Installation

**Prerequisites:** Python 3.9+

```bash
pip install artifacta
```

That's it! The UI is pre-built and bundled. No Node.js required.

### Start Tracking Server

```bash
artifacta ui
```

The web UI will be available at http://localhost:8000 (default).

You can customize host and port:

```bash
artifacta ui --host 0.0.0.0 --port 8000
```

**Development Mode:** Run with hot-reload for UI development:

```bash
artifacta ui --dev
```

### Log Your First Experiment

```python
from artifacta import Series, init, log

# Initialize a run
run = init(
    project="my-project",
    name="experiment-1",
    config={"learning_rate": 0.001, "batch_size": 32}
)

# Log metrics during training
for epoch in range(10):
    train_loss = train_model()  # Your training code

    log("metrics", Series(
        index="epoch",
        fields={
            "train_loss": [train_loss],
            "epoch": [epoch]
        }
    ))

# Log artifacts (models, plots, etc.)
run.log_artifact("model.pt", "path/to/model.pt")
```

---

## Documentation

Full documentation available at: [User Guide](docs/user-guide.rst)

Build and serve docs locally:

```bash
pip install artifacta[dev]
cd docs
make html  # Generates both JSDoc (UI) and Sphinx (Python) docs
python -m http.server 8001 --directory _build/html
```

Then navigate to http://localhost:8001

---

## Core Primitives

Artifacta provides rich primitives for structured logging:

- **Series** - Time series data (loss curves, accuracy over time)
- **Curve** - ROC curves, PR curves with AUC metrics
- **Distribution** - Histograms and distributions
- **Matrix** - Confusion matrices and heatmaps
- **Scatter** - 2D scatter plots (embeddings, parameter spaces)
- **BarChart** - Categorical comparisons
- **Table** - Structured tabular data

All primitives are automatically visualized in the Plots tab.

---

## Web UI Features

- **Plots** - Auto-generated visualizations with multi-run overlay
- **Sweeps** - Hyperparameter analysis with parallel coordinates
- **Artifacts** - File browser with preview for code, images, video, audio
- **Tables** - Metric aggregation and comparison tables
- **Lineage** - Visual artifact provenance graphs
- **Notebooks** - Rich text lab notebook with LaTeX support
- **Chat** - AI assistant for experiment analysis

---

## Examples

Examples are organized by category in [examples/](examples/):

**Core examples** ([examples/core/](examples/core/)):
- Basic tracking - Metrics, parameters, and artifacts
- All primitives - Series, Scatter, Distribution, Matrix, Bar, and more

**ML frameworks** ([examples/ml_frameworks/](examples/ml_frameworks/)):
- PyTorch (MNIST image classification)
- TensorFlow/Keras (regression)
- scikit-learn (classification)
- XGBoost (regression)

**Domain-specific** ([examples/domain_specific/](examples/domain_specific/)):
- A/B testing with statistical analysis
- Protein expression analysis

**Additional domains** - 14 examples in [tests/domains/](tests/domains/):
- Climate, Computer vision, Finance, Genomics, Physics, Robotics, Audio/Video, and more

**Run examples:**

```bash
# Linux/macOS
source venv/bin/activate
python examples/core/01_basic_tracking.py
python examples/ml_frameworks/pytorch_mnist.py
python examples/domain_specific/ab_testing_experiment.py

# Windows (PowerShell)
venv\Scripts\Activate.ps1
python examples/core/01_basic_tracking.py
python examples/ml_frameworks/pytorch_mnist.py
python examples/domain_specific/ab_testing_experiment.py

# Windows (cmd)
venv\Scripts\activate.bat
python examples/core/01_basic_tracking.py
python examples/ml_frameworks/pytorch_mnist.py
python examples/domain_specific/ab_testing_experiment.py
```

---
