.. Artifacta documentation master file

Welcome to Artifacta
=====================

**Artifacta: Universal experiment and artifact tracking — gain insights and optimize models with confidence.**

Architecture
------------

Artifacta consists of three main components:

.. code-block:: text

   ┌─────────────────┐
   │  Python Client  │  (your experiments)
   │   ds.init()     │
   │   ds.log()      │
   └────────┬────────┘
            │
            │ HTTP/REST
            ▼
   ┌─────────────────┐
   │  Backend Server │  (FastAPI)
   │  - Stores runs  │
   │  - Artifacts    │
   │  - Metrics      │
   └────────┬────────┘
            │
            │ HTTP/REST
            ▼
   ┌─────────────────┐
   │   Web UI        │  (React/TypeScript)
   │  - Visualize    │
   │  - Compare      │
   │  - Analyze      │
   └─────────────────┘

Features
--------

- **Rich Primitives**: Log structured data like Series, Distributions, Matrices, Curves, and more
- **ML Framework Integration**: Auto-logging support for PyTorch, TensorFlow, and PyTorch Lightning
- **Interactive Web UI**: Beautiful interface for visualizing and comparing experiments
- **Artifact Management**: Store and version files with automatic metadata extraction
- **Project Organization**: Group related experiments together for easy management
- **Parameter Tracking**: Automatic logging of configurations and hyperparameters

Quick Links
-----------

- :doc:`user-guide` - Complete user guide
- :doc:`examples` - Example notebooks and scripts
- :doc:`api` - Complete API reference
- :doc:`development` - Development and testing guide

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   user-guide
   examples
   api
   development

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
