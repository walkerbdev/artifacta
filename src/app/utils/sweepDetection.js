/**
 * Sweep Detection Utilities
 *
 * Detects valid hyperparameter sweeps across runs for comparison visualization.
 * A valid sweep has:
 * 1. All runs share the same config keys
 * 2. Exactly ONE parameter varies across runs
 * 3. All other parameters are constant
 */

/**
 * Detect if selected runs form a valid hyperparameter sweep
 * @param {Array} runs - Array of run objects with config
 * @returns {Object|null} - Sweep info or null if invalid
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
 * Extract final metrics from a run's structured_data
 * @param {Object} run - Run object
 * @returns {Object} - Map of metric name to final value
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
 * Extract all available metrics across runs
 * Only includes metrics that have valid numeric values in at least one run
 * @param {Array} runs - Runs with extracted metrics
 * @returns {Array} - Array of unique metric names
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
