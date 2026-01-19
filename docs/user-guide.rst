Quick Start Guide
=================

The Problem
-----------

Modern data science and machine learning workflows involve countless experiments—tweaking hyperparameters, adjusting data preprocessing, testing different architectures, updating dependencies, modifying code. **Every change produces different results**, but tracking and comparing these variations manually becomes overwhelming:

- 📋 Which parameters, environment, or code version led to that breakthrough result last week?
- 🔍 How does changing the learning rate affect convergence across multiple runs?
- 📊 What's the actual performance difference between model architectures?
- 🤔 Which preprocessing steps improved accuracy by 2%?
- 🔧 Did upgrading that dependency break model performance?
- 💻 What code changes caused the regression?

Without systematic tracking of **parameters, metrics, code changes, dependencies, and environment**, you're flying blind—relying on scattered notes, terminal output, and memory. **Artifacta solves this** by automatically capturing experiments, configurations, code versions, and artifacts in one place with intelligent visualization.

Ecosystem & Alternatives
------------------------

Artifacta is part of a growing ecosystem of experiment tracking tools. Popular alternatives include:

- `MLflow <https://mlflow.org/>`_ - Open-source platform from Databricks for ML lifecycle management
- `Weights & Biases <https://wandb.ai/>`_ - Cloud-first experiment tracking with team collaboration features
- `Neptune.ai <https://neptune.ai/>`_ - Metadata store for MLOps with extensive integrations
- `Comet ML <https://www.comet.com/>`_ - ML platform with experiment tracking and model production monitoring

**Why Artifacta?** We focus on **automatic visualization discovery**, **domain-agnostic tracking** (not just ML), and **simple self-hosting** with a pre-built UI. No heavy dependencies, no mandatory cloud services—just install and start tracking.

Installation
------------

**Prerequisites:** Python 3.9+

.. code-block:: bash

   pip install artifacta

.. note::
   The UI is pre-built and bundled with the package. No Node.js required.

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

**Prerequisites:** Python 3.9+, Node.js 16+

For contributors who want to modify the source code or UI:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/walkerbdev/artifacta.git
   cd artifacta

   # Build UI from source
   npm install && npm run build

   # Install Python package in editable mode
   pip install -e .

To run examples or tests, install with optional dependencies:

.. code-block:: bash

   pip install -e .[dev]

Starting the Tracking Server
-----------------------------

Production Mode (Default)
~~~~~~~~~~~~~~~~~~~~~~~~~

Start the server with the bundled UI:

.. code-block:: bash

   artifacta ui

The web UI will be available at http://localhost:8000 (default).

You can customize host and port:

.. code-block:: bash

   artifacta ui --host 0.0.0.0 --port 8000

Development Mode
~~~~~~~~~~~~~~~~

For UI development with hot-reload (requires Node.js):

.. code-block:: bash

   artifacta ui --dev

This starts:

- Tracking server on http://localhost:8000 (configurable with ``--port``)
- Vite dev server on http://localhost:5173 (configurable with ``--ui-port``)

The dev server provides hot module replacement for rapid UI development.

Basic Usage
-----------

Here's a simple example to get you started:

.. code-block:: python

   import artifacta as ds

   # Initialize a run
   run = ds.init(
       project="my-project",
       name="experiment-1",
       config={"learning_rate": 0.001, "batch_size": 32}
   )

   # Log metrics during training
   for epoch in range(10):
       train_loss = train_model()  # Your training code

       # Log metrics as a Series
       ds.log("metrics", ds.Series(
           index="epoch",
           fields={
               "train_loss": [train_loss],
               "epoch": [epoch]
           }
       ))

   # Log artifacts (models, plots, etc.)
   run.log_artifact("model.pt", "path/to/model.pt")

   # Run automatically finishes when script exits!

Logging
=======

This section covers what you log from your training scripts using Artifacta.

Logging Metrics
---------------

Artifacta provides rich primitives for logging structured data. These primitives are visualized in the **Plots** tab in the web UI.

**Multi-Run Comparison**: When multiple runs are selected, **Series** and **Curve** primitives are automatically overlaid on the same plot for easy comparison. Other primitives (Matrix, Distribution, Scatter, etc.) are kept separate per run to avoid visual clutter.

**Series** - Time series data (supports multi-run overlay):

.. code-block:: python

   ds.log("training", ds.Series(
       index="step",
       fields={
           "loss": [0.5, 0.3, 0.2],
           "accuracy": [0.6, 0.8, 0.9]
       }
   ))

**Distribution** - Histograms and distributions:

.. code-block:: python

   import numpy as np

   ds.log("weights", ds.Distribution(
       values=np.random.randn(1000)
   ))

**Scatter** - 2D scatter plots:

.. code-block:: python

   ds.log("embeddings", ds.Scatter(
       x=[1, 2, 3, 4],
       y=[2, 4, 6, 8],
       labels=["A", "B", "C", "D"]
   ))

**Matrix** - Confusion matrices and heatmaps:

.. code-block:: python

   ds.log("confusion_matrix", ds.Matrix(
       rows=["True A", "True B"],
       cols=["Class A", "Class B"],
       values=[[10, 2], [3, 15]]
   ))

**Curve** - ROC curves, PR curves, and other X-Y relationships (supports multi-run overlay):

.. code-block:: python

   from sklearn.metrics import roc_curve, auc

   fpr, tpr, _ = roc_curve(y_true, y_scores)
   roc_auc = auc(fpr, tpr)

   ds.log("roc_curve", ds.Curve(
       x=fpr.tolist(),
       y=tpr.tolist(),
       x_label="False Positive Rate",
       y_label="True Positive Rate",
       baseline="diagonal",
       metric={"name": "AUC", "value": float(roc_auc)}
   ))

**BarChart** - Categorical comparisons and model performance:

.. code-block:: python

   ds.log("model_comparison", ds.BarChart(
       categories=["ResNet-50", "EfficientNet-B0", "ViT-Base"],
       groups={
           "accuracy": [0.85, 0.88, 0.90],
           "f1_score": [0.83, 0.86, 0.89]
       },
       x_label="Model Architecture",
       y_label="Score",
       stacked=False
   ))

**Table** - Structured tabular data:

.. code-block:: python

   ds.log("top_variants", ds.Table(
       columns=[
           {"name": "Chromosome", "type": "string"},
           {"name": "Position", "type": "number"},
           {"name": "Type", "type": "string"},
           {"name": "Quality", "type": "number"}
       ],
       data=[
           ["chr1", 12345678, "SNP", 95.2],
           ["chr2", 98765432, "INDEL", 88.7],
           ["chr3", 45678901, "SNP", 92.1]
       ]
   ))

Logging Artifacts
-----------------

Log files like models, datasets, code, and configuration. Artifacts appear in the **Artifacts** tab in the web UI.

.. code-block:: python

   # Log a trained model with artifact metadata
   run.log_artifact("model_checkpoint", "checkpoints/resnet50_best.pt", metadata={
       "framework": "PyTorch",
       "architecture": "ResNet-50",
       "model_size_mb": 45.3,
       "author": "CV Team"
   })

   # Log source code directory (automatically recursive)
   run.log_artifact("training_code", "src/")

Auto-logging Checkpoints
------------------------

Artifacta can automatically log model checkpoints for PyTorch Lightning and TensorFlow:

**PyTorch Lightning:**

.. code-block:: python

   import artifacta as ds
   import pytorch_lightning as pl

   # Enable checkpoint logging
   ds.autolog()

   # Your PyTorch Lightning code works as usual
   trainer = pl.Trainer(max_epochs=10)
   trainer.fit(model, train_loader)

**TensorFlow/Keras:**

.. code-block:: python

   import artifacta as ds
   import tensorflow as tf

   # Enable checkpoint logging
   ds.autolog()

   # Your TensorFlow code works as usual
   model.compile(optimizer='adam', loss='mse')
   model.fit(x_train, y_train, epochs=10)

**Checkpoint Logging Options:**

.. code-block:: python

   # Auto-detect framework and log checkpoints every epoch
   ds.autolog()

   # Disable checkpoint logging
   ds.autolog(log_checkpoints=False)

Note: Autolog captures model checkpoints with metadata. For metrics, use ``ds.log()`` with primitives.

Language-Agnostic Logging
--------------------------

You can interact with the backend directly using HTTP requests. Useful for non-Python environments or custom integrations.

**Health Check:**

.. code-block:: bash

   curl http://127.0.0.1:8000/health

**Create a Run:**

.. code-block:: bash

   curl -X POST http://127.0.0.1:8000/api/runs \
     -H 'Content-Type: application/json' \
     -d '{
       "run_id": "my-run-123",
       "project": "my-project",
       "name": "Experiment 1",
       "config": {"learning_rate": 0.001}
     }'

**Log Metrics (Series):**

.. code-block:: bash

   curl -X POST http://127.0.0.1:8000/api/runs/my-run-123/data \
     -H 'Content-Type: application/json' \
     -d '{
       "name": "training_metrics",
       "primitive_type": "Series",
       "section": "training",
       "data": {
         "index": "step",
         "fields": {
           "loss": [0.5, 0.3, 0.2],
           "step": [1, 2, 3]
         }
       }
     }'

**Register Artifact:**

.. code-block:: bash

   curl -X POST http://127.0.0.1:8000/api/artifacts \
     -H 'Content-Type: application/json' \
     -d '{
       "run_id": "my-run-123",
       "name": "model.pt",
       "hash": "abcd1234",
       "storage_path": "/path/to/model.pt",
       "size_bytes": 1024,
       "metadata": {"framework": "PyTorch"},
       "role": "output"
     }'

**Get Run Details:**

.. code-block:: bash

   curl http://127.0.0.1:8000/api/runs/my-run-123

**List All Runs:**

.. code-block:: bash

   curl 'http://127.0.0.1:8000/api/runs?limit=10'

UI Features
===========

The Artifacta web UI provides several features for visualizing and managing your experiments.

**Important**: To see data in the UI, you must first log it to the database using ``ds.init()`` and ``ds.log()`` in your Python scripts. Then select runs from the **Runs** section in the sidebar.

UI Selection Requirements
--------------------------

Different tabs require different selections to display content:

**Plots Tab** - Requires **at least 1 run selected**

**Sweeps Tab** - Requires **at least 2 runs selected** with same config structure and varying parameters

**Artifacts Tab** - Requires **at least 1 run selected** AND **a file selected from the Files panel in sidebar**

**Tables Tab** - Requires **at least 1 run selected**

**Lineage Tab** - Requires **at least 1 run selected**

**Notebooks Tab** - No selection needed (project-level documentation)

Notebooks Tab
-------------

The **Notebooks** tab provides a rich text editor for documenting experiments:

- **Rich text editing** - Format text with headings, bold, italic, lists, code blocks, and more
- **File attachments** - Upload and preview images, PDFs, audio, video, and code files inline
- **Math equations** - Insert LaTeX equations using ``$...$`` syntax
- **Tables** - Create tables directly in the editor
- **Project organization** - Group notes by project to keep experiments organized

Create and edit notes directly in the web UI using the rich text editor.

.. image:: _static/ELN.gif
   :alt: Notebooks tab with rich text editor
   :align: center
   :width: 100%

|

Plots Tab
---------

The **Plots** tab visualizes all primitives logged via ``ds.log()``:

- **Series charts** - Line plots for time series data (loss, accuracy over epochs) - *supports multi-run overlay*
- **Distributions** - Histograms and distribution plots
- **Scatter plots** - 2D scatter visualizations (embeddings, etc.)
- **Curves** - ROC curves, PR curves with AUC metrics - *supports multi-run overlay*
- **Bar charts** - Model comparisons and categorical data
- **Matrices** - Confusion matrices and heatmaps

Plots are automatically discovered from logged primitives and organized by section. You can drag and resize plots.

**Multi-Run Comparison**: When multiple runs are selected in the sidebar, Series and Curve plots automatically overlay all selected runs on the same chart for easy comparison. Other plot types remain separate per run to avoid visual clutter.

.. image:: _static/Plots.gif
   :alt: Plots tab showing automatic visualization discovery
   :align: center
   :width: 100%

|

.. image:: _static/Plots_2.gif
   :alt: Interactive plots with drag and resize
   :align: center
   :width: 100%

|

Tables Tab
----------

The **Tables** tab displays metrics and data in tabular format:

- **Table primitives** - View structured data from ``ds.Table`` with sortable columns
- **Series aggregations** - View ``ds.Series`` metrics with min/max/final aggregations
- **Run comparison** - Compare multiple runs side-by-side in table format
- **Aggregation modes** - Switch between min, max, or final (last) value for each metric
- **CSV export** - Export table data for further analysis

Sweeps Tab
----------

The **Sweeps** tab analyzes hyperparameter sweeps when you select multiple runs:

- **Parallel coordinates** - Visualize high-dimensional parameter spaces
- **Parameter correlation** - See which hyperparameters most impact metrics
- **Scatter plots** - Plot individual parameters vs target metrics
- **Aggregation options** - Choose last/max/min values for metrics

The tab only appears when selected runs form a valid sweep (same config keys with varying values).

.. image:: _static/Sweeps.gif
   :alt: Sweeps tab for hyperparameter analysis
   :align: center
   :width: 100%

|

Lineage Tab
-----------

The **Lineage** tab shows artifact provenance and dependencies:

- **Visual flow graph** - See which artifacts were inputs/outputs for each run
- **Artifact reuse** - Identify shared artifacts across multiple runs
- **Interactive nodes** - Click nodes to expand and view artifact details (hash, metadata)
- **Connection highlighting** - See relationships between artifacts and runs

This helps track data lineage and understand which datasets or models were used in each experiment.

.. image:: _static/Lineage.gif
   :alt: Lineage tab showing artifact provenance
   :align: center
   :width: 100%

|

Artifacts Tab
-------------

The **Artifacts** tab provides a file browser and preview for logged artifacts:

- **Browse files** - Navigate directory structures from ``run.log_artifact()``
- **Preview content** - View text files, code, CSVs, images, PDFs, audio, and video inline
- **Download files** - Download individual files or entire artifact directories
- **View metadata** - See artifact metadata like size, hash, and custom metadata

Navigate artifacts using the Files panel in the sidebar, then click to preview in this tab.

.. image:: _static/Artifacts_1.gif
   :alt: Artifacts tab file browser and preview
   :align: center
   :width: 100%

|

.. image:: _static/Artifacts_2.gif
   :alt: Artifact preview with code and media files
   :align: center
   :width: 100%

|

Chat Tab
--------

The **Chat** tab provides an AI assistant for analyzing experiment results:

- **LLM integration** - Connect to OpenAI, Anthropic, or local LLMs via LiteLLM
- **Context-aware** - Automatically includes run configs, metrics, and artifact contents
- **Code analysis** - Analyzes logged code artifacts to provide insights
- **Interactive Q&A** - Ask questions about your experiments and get recommendations

Configure your LLM API key and model in the settings (gear icon) to start chatting.

.. image:: _static/Chat.gif
   :alt: Chat tab with AI assistant
   :align: center
   :width: 100%

|
