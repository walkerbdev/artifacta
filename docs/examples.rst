Examples
========

Artifacta includes comprehensive examples demonstrating various use cases and logging capabilities.

All examples are located in the `examples/` directory and can be run standalone.

Complete Examples
-----------------

PyTorch MNIST Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File:** `examples/pytorch_mnist.py <https://github.com/walkerbdev/artifacta/blob/main/examples/pytorch_mnist.py>`_

Demonstrates PyTorch integration with Artifacta by running **3 experiments** with different hyperparameters:

1. **Low LR + Adam**: learning_rate=0.001, optimizer=Adam
2. **Medium LR + SGD**: learning_rate=0.01, optimizer=SGD
3. **High LR + SGD**: learning_rate=0.05, optimizer=SGD

Each experiment logs:

- **ds.autolog()** - Automatic checkpoint logging
- **ds.Series** - Track training/validation loss and accuracy curves
- **ds.Matrix** - Log confusion matrix for classification performance
- **Artifact logging** - Save trained PyTorch model with metadata

**Run:**

.. code-block:: bash

   python examples/pytorch_mnist.py

**What you'll see in the UI:**
- Training curves for all 3 runs in the Plots tab (compare optimizers/learning rates)
- Confusion matrices for each run
- Model checkpoints in the Artifacts tab
- Run comparison in the Tables tab

TensorFlow Regression
~~~~~~~~~~~~~~~~~~~~~

**File:** `examples/tensorflow_regression.py <https://github.com/walkerbdev/artifacta/blob/main/examples/tensorflow_regression.py>`_

Demonstrates TensorFlow/Keras integration by running **3 experiments** with different network architectures:

1. **Small Network**: hidden_dim=32, learning_rate=0.001
2. **Medium Network**: hidden_dim=64, learning_rate=0.01
3. **Large Network**: hidden_dim=128, learning_rate=0.001

Each experiment logs:

- **ds.autolog()** - Automatic Keras checkpoint logging
- **ds.Series** - Track loss and MAE over epochs
- **ds.Scatter** - Visualize predictions vs actual values
- **ds.Distribution** - Analyze prediction residuals
- **Artifact logging** - Save Keras model with metadata

**Run:**

.. code-block:: bash

   python examples/tensorflow_regression.py

**What you'll see in the UI:**
- Training/validation loss curves for all 3 network sizes
- Scatter plots comparing predictions across architectures
- Residual distribution histograms
- R² scores and error metrics for model comparison

A/B Testing Experiment (Domain-Agnostic)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File:** `examples/ab_testing_experiment.py <https://github.com/walkerbdev/artifacta/blob/main/examples/ab_testing_experiment.py>`_

**Artifacta isn't just for ML!** This example demonstrates experiment tracking for A/B testing:

- Simulate A/B test for e-commerce button colors
- **Parameter sweep** - Run multiple experiments with different sample sizes
- **ds.Distribution** - Compare conversion rates across variants
- **ds.Series** - Track cumulative conversions over time
- **ds.BarChart** - Visualize performance comparison
- Statistical significance testing and lift calculations

**Run:**

.. code-block:: bash

   python examples/ab_testing_experiment.py

**What you'll see in the UI:**
- Conversion rate distributions by variant (Control, Green, Red)
- Time series of cumulative conversions
- Statistical significance and lift percentages
- Sweeps tab showing how sample size affects confidence

**Scenario:** Testing checkout button colors (Blue vs Green vs Red) to maximize conversions. Shows that Artifacta works for ANY parametric experiment tracking - not just machine learning!

Running the Examples
--------------------

**1. Install Artifacta with all dependencies:**

.. code-block:: bash

   pip install -e '.[dev]'

This installs Artifacta and all optional dependencies including PyTorch, TensorFlow, and scientific computing libraries from the ``pyproject.toml`` file.

**2. Activate the virtual environment:**

.. code-block:: bash

   source venv/bin/activate

**3. Start the tracking server:**

.. code-block:: bash

   artifacta ui

The web UI will be available at the URL shown in the terminal output.

**4. Run examples** (in a separate terminal with venv activated):

.. code-block:: bash

   source venv/bin/activate  # Activate in the new terminal too
   python examples/pytorch_mnist.py
   python examples/tensorflow_regression.py
   python examples/ab_testing_experiment.py

**5. View results in the web UI:**

- **Plots tab** - Interactive visualizations of all logged metrics
- **Tables tab** - Tabular view with aggregations
- **Artifacts tab** - Browse and preview saved models
- **Sweeps tab** - Analyze parameter sweeps (A/B testing example)
- **Notebooks tab** - Document your findings
