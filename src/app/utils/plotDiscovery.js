/**
 * Universal plot discovery from structured data primitives
 *
 * This module implements automatic plot discovery - analyzing logged data primitives
 * and determining the appropriate visualizations to display. It's the core algorithm
 * that powers the "Plots" tab, automatically creating charts without manual configuration.
 *
 * Primitive → Plot Type Mapping:
 * - Series → Line plots, scatter plots, multi-run overlays
 * - Distribution → Histograms, violin plots
 * - Matrix → Heatmaps
 * - Scatter → Scatter plots (2D point clouds)
 * - Curve → ROC curves, PR curves (with AUC metrics)
 * - BarChart → Bar charts (categorical comparisons)
 * - Table → Data tables
 *
 * Multi-run support:
 * - Automatically detects when multiple runs log the same primitive
 * - Creates overlay plots (multiple series on same chart)
 * - Handles both single-run and multi-run cases transparently
 */

/**
 * Discover all plots from structured data, grouped by section
 *
 * This is the main entry point for plot discovery. It analyzes all logged primitives
 * in a run (or across multiple runs) and generates plot configurations that can be
 * rendered by visualization components.
 *
 * Algorithm:
 * 1. Iterate through all structured data entries (keyed by primitive name)
 * 2. Detect if data is single-run or multi-run (check for _runId marker)
 * 3. For each primitive:
 *    a. Extract primitive type (series, curve, distribution, etc.)
 *    b. Map to appropriate plot type(s) via mapPrimitiveToPlots()
 *    c. Transform data to plot-specific format
 *    d. Group by section (user-specified or "General")
 * 4. Return section-organized plot configs ready for rendering
 *
 * Single-run vs Multi-run:
 * - Single-run: Takes most recent entry for each primitive
 * - Multi-run: Combines all runs' data into overlay plots
 *
 * @param {object} structuredData - Object mapping primitive names to data entries
 *   Format: { "primitiveName": [{ primitive_type, section, data, metadata, timestamp }] }
 *   Multi-run format: entries have _runId, _runName fields
 * @returns {object} Plots organized by section
 *   Format: { "sectionName": [{ plotId, plotType, title, data, ... }] }
 *
 * @example
 * const structuredData = {
 *   "metrics::loss": [{
 *     primitive_type: "series",
 *     section: "Training",
 *     data: { index: "epoch", fields: { loss: [0.5, 0.3, 0.2] } }
 *   }]
 * };
 * const plots = discoverPlots(structuredData);
 * // Returns: {
 * //   "Training": [{
 * //     plotId: "metrics::loss-line",
 * //     plotType: "line",
 * //     title: "Loss",
 * //     data: { series: [...] }
 * //   }]
 * // }
 */
export function discoverPlots(structuredData) {
  if (!structuredData) return {};

  const plotsBySection = {};

  Object.entries(structuredData).forEach(([key, entries]) => {
    // Each entry is {primitive_type, section, data, metadata, timestamp}
    // For multi-run primitives (series/curve), entries is array of all runs' data
    // For single-run primitives, entries is just that run's data

    // Check if this is multi-run data (has _runId marker)
    const isMultiRun = entries.length > 0 && entries[0]._runId !== undefined;

    if (isMultiRun) {
      // Multi-run data: all entries are from different runs but same primitive
      // Use first entry to get primitive type and section
      const firstEntry = entries[0];
      const { primitive_type, section } = firstEntry;

      // Extract original name from key (format: "section::name" or just "name")
      const name = key.includes('::') ? key.split('::')[1] : key;

      // Map to plot configs (handles multi-run internally)
      const plotConfigs = mapPrimitiveToPlots(name, primitive_type, entries, null, isMultiRun);

      // Group by section
      const sectionName = section || 'General';
      if (!plotsBySection[sectionName]) {
        plotsBySection[sectionName] = [];
      }
      plotsBySection[sectionName].push(...plotConfigs);
    } else {
      // Single-run data: use most recent entry
      const latestEntry = entries[entries.length - 1];
      const { primitive_type, section, data, metadata } = latestEntry;

      // For multi-run selection, keep the full key (with run prefix) to ensure unique IDs
      // For single-run, extract the name without prefix
      const name = key;

      // Map primitive to plot types
      const plotConfigs = mapPrimitiveToPlots(name, primitive_type, data, metadata, false);

      // Group by section
      const sectionName = section || 'General';
      if (!plotsBySection[sectionName]) {
        plotsBySection[sectionName] = [];
      }
      plotsBySection[sectionName].push(...plotConfigs);
    }
  });

  return plotsBySection;
}

/**
 * Map a primitive to one or more plot configs.
 * @param {string} name - Primitive name
 * @param {string} primitiveType - Type of primitive
 * @param {unknown} data - Either single-run data or array of multi-run entries
 * @param {object} metadata - Metadata (for single-run)
 * @param {boolean} isMultiRun - Whether this is multi-run data
 * @returns {Array<object>} Array of plot configuration objects
 */
function mapPrimitiveToPlots(name, primitiveType, data, metadata, isMultiRun = false) {
  const plots = [];

  switch (primitiveType) {
    case 'series':
      if (isMultiRun) {
        // Multi-run series: data is array of entries with _runName, _runId
        // All runs are overlaid on the same chart, so use a single plot ID
        plots.push({
          id: `${name}_line`,
          type: 'line',
          title: formatTitle(name),
          data: transformMultiRunSeriesForLinePlot(data),
          metadata: data[0].metadata // Use metadata from first run
        });
      } else {
        // Single-run series
        plots.push({
          id: `${name}_line`,
          type: 'line',
          title: formatTitle(name),
          data: transformSeriesForLinePlot(data),
          metadata
        });
      }
      break;

    case 'distribution':
      plots.push({
        id: `${name}_histogram`,
        type: 'histogram',
        title: formatTitle(name),
        data: transformDistributionForHistogram(data),
        metadata
      });

      // If grouped, also create violin plot
      if (data.groups) {
        plots.push({
          id: `${name}_violin`,
          type: 'violin',
          title: `${formatTitle(name)} by group`,
          data: transformDistributionForViolin(data),
          metadata
        });
      }
      break;

    case 'matrix':
      plots.push({
        id: `${name}_heatmap`,
        type: 'heatmap',
        title: formatTitle(name),
        data: data,
        metadata
      });
      break;


    case 'graph':
      break;

    case 'table':
      // Table primitive is rendered in Tables tab, not as a plot
      break;

    case 'events':
      // Events/timelines are not useful for parameter sweeps
      // (same events across runs, different absolute timestamps)
      // Users should log durations as scalars if timing is important
      break;

    case 'media':
      // Media (images/videos) should be logged as dataset artifacts
      // and viewed in the Datasets tab, not as plots
      break;

    case 'curve':
      if (isMultiRun) {
        // Multi-run curve: overlay multiple ROC/PR curves
        plots.push({
          id: `${name}_curve`,
          type: 'curve',
          title: formatTitle(name),
          data: transformMultiRunCurveForCurveChart(data),
          metadata: data[0].metadata // Use metadata from first run
        });
      } else {
        // Single-run curve
        plots.push({
          id: `${name}_curve`,
          type: 'curve',
          title: formatTitle(name),
          data: transformCurveForCurveChart(data),
          metadata
        });
      }
      break;

    case 'scatter':
      plots.push({
        id: `${name}_scatter`,
        type: 'scatter',
        title: formatTitle(name),
        data: transformScatterForScatterPlot(data),
        metadata
      });
      break;

    case 'barchart':
      plots.push({
        id: `${name}_barchart`,
        type: 'barchart',
        title: formatTitle(name),
        data: data,
        metadata
      });
      break;

    case 'field':
      break;

    case 'hierarchy':
      break;
  }

  return plots;
}

/**
 * Format name for display (snake_case -> Title Case).
 * @param {string} name - Name to format
 * @returns {string} Formatted title string
 */
function formatTitle(name) {
  return name
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

/**
 * Transform Series to line plot format.
 * @param {object} seriesData - Series data with index, fields, and index_values
 * @returns {object} Transformed data for line plot
 */
function transformSeriesForLinePlot(seriesData) {
  const { index, fields, index_values } = seriesData;

  // Create x-axis values
  const xValues = index_values || Array.from({ length: Object.values(fields)[0].length }, (_, i) => i);

  // Create dataset for each field
  const datasets = Object.entries(fields).map(([fieldName, values]) => ({
    label: fieldName,
    data: xValues.map((x, i) => ({ x, y: values[i] }))
  }));

  return {
    xLabel: index,
    datasets
  };
}

/**
 * Transform Distribution to histogram format.
 * @param {object} distData - Distribution data with values, groups, and bins
 * @returns {object} Transformed data for histogram
 */
function transformDistributionForHistogram(distData) {
  return {
    values: distData.values,
    groups: distData.groups,
    bins: 30  // default, can be overridden by metadata
  };
}

/**
 * Transform Distribution to violin plot format.
 * @param {object} distData - Distribution data with values and groups
 * @returns {object} Transformed data for violin plot
 */
function transformDistributionForViolin(distData) {
  // Group values by group label
  const groupedData = {};
  distData.values.forEach((value, i) => {
    const group = distData.groups[i];
    if (!groupedData[group]) groupedData[group] = [];
    groupedData[group].push(value);
  });

  return {
    groups: Object.keys(groupedData),
    data: Object.values(groupedData)
  };
}

/**
 * Transform Curve to CurveChart format.
 * @param {object} curveData - Curve data with x, y, labels, baseline, and metric
 * @returns {object} Transformed data for curve chart
 */
function transformCurveForCurveChart(curveData) {
  const { x, y, x_label, y_label, baseline, metric } = curveData;

  // Convert x, y arrays to points array
  const points = x.map((xVal, i) => ({
    x: xVal,
    y: y[i]
  }));

  return {
    data: points,
    xLabel: x_label,
    yLabel: y_label,
    showDiagonal: baseline === 'diagonal',
    metric: metric ? metric.value : undefined,
    metricLabel: metric ? metric.name : undefined
  };
}

/**
 * Transform Scatter to ScatterPlot format.
 * @param {object} scatterData - Scatter data with points, x_label, and y_label
 * @returns {object} Transformed data for scatter plot
 */
function transformScatterForScatterPlot(scatterData) {
  const { points, x_label, y_label } = scatterData;

  if (!points || points.length === 0) {
    return { x: [], y: [], label: 'Scatter', xLabel: x_label, yLabel: y_label };
  }

  // Auto-detect field names from first point (don't assume "x" and "y")
  const firstPoint = points[0];
  const numericFields = Object.keys(firstPoint).filter(key => {
    const val = firstPoint[key];
    return typeof val === 'number' && !isNaN(val);
  });

  if (numericFields.length < 2) {
    console.warn('[plotDiscovery] Scatter points need at least 2 numeric fields, found:', numericFields);
    return { x: [], y: [], label: 'Scatter', xLabel: x_label, yLabel: y_label };
  }

  const xField = numericFields[0];
  const yField = numericFields[1];

  // Extract arrays using detected field names
  const x = points.map(p => p[xField]);
  const y = points.map(p => p[yField]);

  return {
    x,
    y,
    label: x_label && y_label ? `${y_label} vs ${x_label}` : 'Scatter',
    xLabel: x_label,
    yLabel: y_label
  };
}

/**
 * Transform multi-run Series to line plot format
 * Overlays all runs on same chart with different colors.
 * @param {Array<object>} runEntries - Array of run entries with data, _runName, and _runId
 * @returns {object} Transformed data for multi-run line plot
 */
function transformMultiRunSeriesForLinePlot(runEntries) {
  const allDatasets = [];

  // Each entry has: {data: {index, fields, index_values}, _runName, _runId}
  runEntries.forEach(entry => {
    const { fields, index_values } = entry.data;
    const xValues = index_values || Array.from({ length: Object.values(fields)[0].length }, (_, i) => i);

    // Create datasets for this run, labeled with run name
    Object.entries(fields).forEach(([fieldName, values]) => {
      allDatasets.push({
        label: `${entry._runName}: ${fieldName}`,
        data: xValues.map((x, i) => ({ x, y: values[i] }))
      });
    });
  });

  return {
    xLabel: runEntries[0].data.index,
    datasets: allDatasets
  };
}

/**
 * Transform multi-run Curve to curve chart format
 * Overlays multiple ROC/PR curves with different colors.
 * @param {Array<object>} runEntries - Array of run entries with curve data
 * @returns {object} Transformed data for multi-run curve chart
 */
function transformMultiRunCurveForCurveChart(runEntries) {
  // For curves, we need to return multiple curve datasets
  // CurveChart component will need to be updated to handle this
  // For now, return format similar to single curve but with multiple series

  const curves = runEntries.map(entry => {
    const { x, y, metric } = entry.data;

    return {
      label: entry._runName,
      points: x.map((xVal, i) => ({ x: xVal, y: y[i] })),
      metric: metric ? metric.value : undefined
    };
  });

  // Use first entry for common properties
  const firstEntry = runEntries[0].data;

  return {
    curves, // Array of curves with labels
    xLabel: firstEntry.x_label,
    yLabel: firstEntry.y_label,
    showDiagonal: firstEntry.baseline === 'diagonal',
    metricLabel: firstEntry.metric ? firstEntry.metric.name : undefined
  };
}
