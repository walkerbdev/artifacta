UI Components & Utilities API
==============================

This section documents the React components, hooks, and utility functions that power the Artifacta user interface.

.. note::
   The UI is built with React and uses HTML5 Canvas for high-performance plot rendering.
   All visualizations are generated server-side from data primitives logged via the Python API.

Organization
------------

The UI codebase is organized into:

**Components**
  React components for visualization, tabs, layout, and user interaction

**Hooks**
  Custom React hooks for data fetching, canvas rendering, and state management

**Utils**
  Utility functions for automatic plot discovery, data processing, and aggregation

**Pages**
  Top-level page components and routing

Key UI Features
---------------

Tabs
~~~~

The Artifacta UI provides several specialized tabs for experiment analysis:

- **Plots Tab** - Automatically discovered visualizations from logged primitives
- **Sweeps Tab** - Hyperparameter analysis with parallel coordinates and correlation plots
- **Artifacts Tab** - File browser with rich media preview (video, audio, PDF, code)
- **Lineage Tab** - Interactive artifact provenance graph
- **Chat Tab** - AI assistant for experiment analysis
- **Project Notes** - Electronic lab notebook with LaTeX and file attachments

Plot Components
~~~~~~~~~~~~~~~

All plot types support interactive tooltips and are rendered on HTML5 Canvas for performance:

- **LinePlot** - Time series with multi-run overlay support
- **ScatterPlot** - 2D scatter with nearest-point tooltips
- **Heatmap** - Matrix visualization with cell-level tooltips
- **CurveChart** - ROC/PR curves with AUC calculation
- **ParallelCoordinatesChart** - Hyperparameter relationships
- **BarChart** - Categorical comparisons
- **Histogram** - Distribution visualization
- **ViolinPlot** - Distribution comparison across categories

Core Utilities
~~~~~~~~~~~~~~

**Plot Discovery** (``plotDiscovery.js``)
  Automatically generates appropriate visualizations from data primitive types:

  - Series → LinePlot
  - Distribution → Histogram/ViolinPlot
  - Matrix → Heatmap
  - Curve → CurveChart with AUC
  - Scatter → ScatterPlot
  - BarChart → BarChart

**Multi-Run Comparison** (``comparisonPlotDiscovery.js``)
  Handles overlay logic for comparing multiple experiment runs on the same plot.
  Supports LinePlot and CurveChart overlay modes.

**Metric Aggregation** (``metricAggregation.js``)
  Aggregates metrics across runs for analysis views (mean, std, min, max, latest).

**Sweep Detection** (``sweepDetection.js``)
  Identifies hyperparameter sweeps from run configurations and generates
  parallel coordinates and correlation visualizations.

Custom Hooks
~~~~~~~~~~~~

**useCanvasTooltip** (``hooks/useCanvasTooltip.js``)
  Manages interactive tooltips for canvas-based plots. Provides 60fps tooltip
  updates using requestAnimationFrame.

  Supported tooltip types:

  - ``series`` - Multi-series line plots
  - ``scatter`` - Scatter plot points
  - ``matrix`` - Heatmap cells
  - ``curve`` - ROC/PR curve points

**useResponsiveCanvas** (``hooks/useResponsiveCanvas.js``)
  Handles responsive canvas sizing with HiDPI (Retina) support.
  Automatically adjusts canvas resolution for crisp rendering.

**useRunData** (``hooks/useRunData.js``)
  Fetches and caches experiment run data from the tracking server.
  Handles multi-run selection and data aggregation.

**useLayoutManager** (``hooks/useLayoutManager.js``)
  Manages drag-and-drop layout persistence for plots and visualizations.
  Stores layout preferences in browser localStorage.

Full API Documentation
----------------------

For complete API documentation with function signatures, parameters, and return types,
see the auto-generated JSDoc documentation:

.. raw:: html

   <p><a href="../jsdoc/index.html" target="_blank" class="reference external">
   View Full UI API Documentation (JSDoc) →
   </a></p>

.. note::
   The JSDoc documentation is generated from inline comments in the source code
   and opens in a new window. It provides detailed information about all components,
   hooks, utilities, and their usage.

Building the UI Docs
--------------------

To regenerate the JSDoc documentation::

    npm run docs:ui

This will scan the ``src/app`` directory and generate HTML documentation in ``docs/_build/jsdoc/``.
