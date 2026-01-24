Examples
========

Artifacta includes comprehensive examples demonstrating all features and use cases.

All examples are located in the `examples/ directory <https://github.com/walkerbdev/artifacta/tree/main/examples>`_.

Installation
------------

Install Artifacta with all dependencies needed to run the examples:

.. code-block:: bash

   # Create a new virtual environment
   python3 -m venv artifacta-examples
   source artifacta-examples/bin/activate  # On Windows: artifacta-examples\Scripts\activate

   # Install artifacta + ML frameworks (sklearn, xgboost, pytorch, tensorflow)
   pip install -r examples/requirements.txt

Quick Start
-----------

**Start the UI server** (in a separate terminal):

.. code-block:: bash

   artifacta ui
   # View at http://localhost:8000

**Run the minimal example:**

.. code-block:: bash

   python examples/core/01_basic_tracking.py

**Run all examples:**

.. code-block:: bash

   python examples/run_all_examples.py

Core Examples
-------------

Basic Tracking
~~~~~~~~~~~~~~

**File:** `examples/core/01_basic_tracking.py <https://github.com/walkerbdev/artifacta/blob/main/examples/core/01_basic_tracking.py>`_

Minimal "hello world" example showing the basic workflow:

- Initialize a run with ``init()``
- Log metrics with ``Series`` primitive
- Auto-finish behavior (no need to call ``finish()``)

**Run:** ``python examples/core/01_basic_tracking.py``

All Primitives Demo
~~~~~~~~~~~~~~~~~~~

**File:** `examples/core/02_all_primitives.py <https://github.com/walkerbdev/artifacta/blob/main/examples/core/02_all_primitives.py>`_

Comprehensive demo of all 7 data primitives:

- **Series** - Time series data
- **Distribution** - Histograms
- **Matrix** - 2D heatmaps
- **Table** - Structured data
- **Curve** - X/Y relationships
- **Scatter** - Point clouds
- **BarChart** - Categorical comparisons

**Run:** ``python examples/core/02_all_primitives.py``

ML Framework Examples
---------------------

Sklearn Classification
~~~~~~~~~~~~~~~~~~~~~~

**File:** `examples/ml_frameworks/sklearn_classification.py <https://github.com/walkerbdev/artifacta/blob/main/examples/ml_frameworks/sklearn_classification.py>`_

Binary classification with RandomForestClassifier:

- ``autolog()`` integration for sklearn
- ROC and Precision-Recall curves
- Confusion matrix
- Feature importance

**Run:** ``python examples/ml_frameworks/sklearn_classification.py``

XGBoost Regression
~~~~~~~~~~~~~~~~~~

**File:** `examples/ml_frameworks/xgboost_regression.py <https://github.com/walkerbdev/artifacta/blob/main/examples/ml_frameworks/xgboost_regression.py>`_

Hyperparameter grid search with XGBoost:

- Multiple runs with different configs
- Feature importance tracking
- Prediction scatter plots
- Model comparison

**Run:** ``python examples/ml_frameworks/xgboost_regression.py``

PyTorch MNIST
~~~~~~~~~~~~~

**File:** `examples/ml_frameworks/pytorch_mnist.py <https://github.com/walkerbdev/artifacta/blob/main/examples/ml_frameworks/pytorch_mnist.py>`_

MNIST digit classification with PyTorch:

- ``autolog()`` integration for PyTorch
- Training/validation curves
- Confusion matrix
- Model checkpoints

**Run:** ``python examples/ml_frameworks/pytorch_mnist.py``

TensorFlow Regression
~~~~~~~~~~~~~~~~~~~~~

**File:** `examples/ml_frameworks/tensorflow_regression.py <https://github.com/walkerbdev/artifacta/blob/main/examples/ml_frameworks/tensorflow_regression.py>`_

Regression with TensorFlow/Keras:

- ``autolog()`` integration for TensorFlow
- Loss and MAE curves
- Prediction scatter plots
- Model saving

**Run:** ``python examples/ml_frameworks/tensorflow_regression.py``

Domain-Specific Examples
-------------------------

A/B Testing Experiment
~~~~~~~~~~~~~~~~~~~~~~

**File:** `examples/domain_specific/ab_testing_experiment.py <https://github.com/walkerbdev/artifacta/blob/main/examples/domain_specific/ab_testing_experiment.py>`_

**Artifacta isn't just for ML!** This example demonstrates A/B testing:

- Simulate e-commerce button color test
- Parameter sweep with different sample sizes
- Conversion rate distributions
- Statistical significance testing

**Run:** ``python examples/domain_specific/ab_testing_experiment.py``

Protein Expression Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File:** `examples/domain_specific/protein_expression.py <https://github.com/walkerbdev/artifacta/blob/main/examples/domain_specific/protein_expression.py>`_

Wet lab experiment tracking for biology:

- Factorial design (temperature × IPTG × time)
- Yield, purity, activity measurements
- Growth curves
- Parameter correlation analysis

**Run:** ``python examples/domain_specific/protein_expression.py``
