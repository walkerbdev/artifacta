/**
 * Sweep Detection Utilities
 *
 * Detects valid hyperparameter sweeps across experiment runs for comparison visualization.
 *
 * A sweep is a set of runs where hyperparameters are systematically varied to understand
 * their effect on metrics. This module validates sweep structure and extracts metadata
 * needed for sweep visualization (parallel coordinates, scatter plots, correlation analysis).
 *
 * Validation criteria:
 * 1. All runs must have a config object
 * 2. All runs share the same config keys (consistent structure)
 * 3. At least one parameter varies across runs
 * 4. At least 2 runs (need comparison)
 *
 * The module also extracts final metrics from each run's structured_data for comparison.
 */

/**
 * Detect if selected runs form a valid hyperparameter sweep
 *
 * Algorithm:
 * 1. Filter runs that have non-empty config
 * 2. Verify all runs have identical config key structure
 * 3. For each config key, check if values vary across runs
 * 4. Classify parameters as varying (different values) or constant (same value)
 * 5. Extract final metric values from each run's structured_data
 * 6. Return sweep metadata including varying params, metrics, and transformed runs
 *
 * A sweep is valid if:
 * - At least 2 runs with config
 * - All configs have same keys
 * - At least 1 parameter varies
 *
 * @param {Array<object>} runs - Array of run objects, each with:
 *   - run_id: unique identifier
 *   - name: run name
 *   - config: object with hyperparameters
 *   - structured_data: logged metrics/plots
 * @returns {object|null} Sweep detection result:
 *   - If valid: { valid: true, varyingParams, constantParams, runs, availableMetrics, ... }
 *   - If invalid: { valid: false, reason, message }
 *   - If insufficient runs: null
 *
 * @example
 * const runs = [
 *   { config: { lr: 0.01, batch: 32 }, structured_data: {...} },
 *   { config: { lr: 0.1,  batch: 32 }, structured_data: {...} },
 *   { config: { lr: 1.0,  batch: 32 }, structured_data: {...} }
 * ];
 * const sweep = detectSweep(runs);
 * // Returns: {
 * //   valid: true,
 * //   varyingParams: [{ name: 'lr', values: [0.01, 0.1, 1.0], isNumeric: true }],
 * //   constantParams: [{ name: 'batch', value: 32 }],
 * //   runs: [...],  // enriched with metrics
 * //   availableMetrics: ['loss', 'accuracy']
 * // }
 */
export function detectSweep(runs) {
  if (!runs || runs.length < 2) {
    return null;
  }

  // Filter runs that have config
  const runsWithConfig = runs.filter(r => r.config && Object.keys(r.config).length > 0);

  if (runsWithConfig.length < 2) {
    return null;
  }

  // Check if all runs have same config keys
  const firstKeys = Object.keys(runsWithConfig[0].config).sort();

  const allSameKeys = runsWithConfig.every(run => {
    const keys = Object.keys(run.config).sort();
    return keys.length === firstKeys.length &&
           keys.every((key, i) => key === firstKeys[i]);
  });

  if (!allSameKeys) {
    return {
      valid: false,
      reason: 'inconsistent_keys',
      message: 'Selected runs have different hyperparameter keys. All runs must have the same config structure.'
    };
  }

  // Find which parameters vary
  const varyingParams = [];
  const constantParams = [];

  firstKeys.forEach(key => {
    const values = runsWithConfig.map(r => r.config[key]);
    const uniqueValues = new Set(values.map(v => JSON.stringify(v)));

    if (uniqueValues.size > 1) {
      const parsedValues = Array.from(uniqueValues).map(v => JSON.parse(v));
      const isNumeric = parsedValues.every(v => typeof v === 'number');

      varyingParams.push({
        name: key,
        values: parsedValues,
        isNumeric
      });
    } else {
      constantParams.push({
        name: key,
        value: values[0]
      });
    }
  });

  // Valid sweep: at least one parameter varies
  if (varyingParams.length === 0) {
    return {
      valid: false,
      reason: 'no_varying_params',
      message: 'All hyperparameters are identical across runs. No sweep detected.',
      varyingParams,
      constantParams
    };
  }

  // Get final metrics from last logged data
  const runsWithMetrics = runsWithConfig.map(run => {
    const metrics = extractFinalMetrics(run);
    return {
      run_id: run.run_id,
      name: run.name,
      config: run.config,
      metrics
    };
  });

  const availableMetrics = extractAvailableMetrics(runsWithMetrics);

  return {
    valid: true,
    varyingParams: varyingParams,  // Now supports multiple varying parameters
    sweptParameter: varyingParams[0].name,  // Kept for backwards compatibility
    sweptValues: varyingParams[0].values,   // Kept for backwards compatibility
    constantParams,
    runs: runsWithMetrics,
    availableMetrics
  };
}

/**
 * Extract final metric values from a run's structured_data
 *
 * Searches through all structured data entries (Series, Curve primitives) and extracts
 * the final/summary value for each metric. This is used to create a single metric
 * value per run for sweep comparison.
 *
 * Extraction logic:
 * - Series primitives: Takes the LAST value from each field array (final epoch value)
 * - Curve primitives: Extracts the summary metric (e.g., AUC from ROC curve)
 *
 * @param {object} run - Run object with structured_data property
 * @returns {object} Map of metric names to final numeric values
 *   Example: { loss: 0.23, accuracy: 0.95, auc: 0.88 }
 */
function extractFinalMetrics(run) {
  const metrics = {};

  if (!run.structured_data) {
    return metrics;
  }

  Object.entries(run.structured_data).forEach(([_name, entries]) => {
    if (!entries || entries.length === 0) return;

    const latestEntry = entries[entries.length - 1];
    const data = latestEntry.data;

    // Extract scalar metrics from series
    if (latestEntry.primitive_type === 'series' && data.fields) {
      Object.entries(data.fields).forEach(([fieldName, values]) => {
        if (Array.isArray(values) && values.length > 0) {
          // Use last value as final metric
          metrics[fieldName] = values[values.length - 1];
        }
      });
    }

    // Extract metrics from curve (e.g., AUC from ROC curve)
    if (latestEntry.primitive_type === 'curve' && data.metric) {
      metrics[data.metric.name] = data.metric.value;
    }
  });

  return metrics;
}

/**
 * Extract all available metric names across multiple runs
 *
 * Collects the union of all metric names that have valid numeric values in at least
 * one run. Filters out null, undefined, NaN, and non-numeric values.
 *
 * This is used to populate metric selector dropdowns in sweep visualizations, ensuring
 * users only see metrics that actually have data.
 *
 * @param {Array<object>} runs - Runs with extracted metrics (output from extractFinalMetrics)
 *   Each run should have shape: { metrics: { metricName: numericValue, ... } }
 * @returns {Array<string>} Sorted array of unique metric names that have valid numeric data
 *
 * @example
 * const runs = [
 *   { metrics: { loss: 0.5, accuracy: 0.9 } },
 *   { metrics: { loss: 0.3, f1: 0.85 } }
 * ];
 * extractAvailableMetrics(runs);
 * // Returns: ['accuracy', 'f1', 'loss']  (sorted alphabetically)
 */
function extractAvailableMetrics(runs) {
  const allMetrics = new Set();

  runs.forEach(run => {
    if (run.metrics) {
      Object.entries(run.metrics).forEach(([metric, value]) => {
        // Only include metrics with valid numeric values
        if (typeof value === 'number' && !isNaN(value)) {
          allMetrics.add(metric);
        }
      });
    }
  });

  return Array.from(allMetrics).sort();
}
