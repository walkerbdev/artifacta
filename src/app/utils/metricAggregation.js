/**
 * Client-side metric aggregation utilities
 *
 * Provides generic aggregation functions for extracting summary statistics from
 * metrics logged across training runs. Designed to be fully data-structure agnostic,
 * working with any x-axis field (epoch, step, timestamp, etc.) and any metric names.
 *
 * Key capabilities:
 * - Auto-detect x-axis fields (epoch, step, etc.)
 * - Aggregate metrics (min, max, final, best)
 * - Extract metrics from Series primitives in structured_data
 * - Find optimal values based on optimization metrics (e.g., accuracy at min loss)
 * - Discover available metrics across runs
 *
 * Used primarily by the Tables tab for displaying run comparisons.
 */

/**
 * Detect x-axis field from metrics array using heuristics
 *
 * Finds integer-valued fields that could represent the x-axis (epoch, step, iteration).
 * If multiple candidates exist, prefers smaller values (epoch over timestamp).
 *
 * Algorithm:
 * 1. Find all integer-valued fields (excludes internal fields starting with _)
 * 2. If only one integer field, use it
 * 3. If multiple, prefer field with smallest value (epoch=10 over timestamp=1674...)
 *
 * @param {Array<object>} metricsArray - Array of metric records
 *   Example: [{ epoch: 0, loss: 0.5 }, { epoch: 1, loss: 0.3 }]
 * @returns {string|null} X-axis field name (e.g., "epoch", "step") or null if none found
 */
const detectXAxisField = (metricsArray) => {
  if (!metricsArray || metricsArray.length === 0) return null;

  const sample = metricsArray[0];
  const integerFields = Object.keys(sample).filter(key =>
    !key.startsWith('_') && Number.isInteger(sample[key])
  );

  if (integerFields.length === 0) return null;
  if (integerFields.length === 1) return integerFields[0];

  // Prefer smaller values (epoch/step over timestamp)
  const fieldValues = integerFields.map(field => ({
    field,
    value: sample[field]
  }));
  fieldValues.sort((a, b) => a.value - b.value);
  return fieldValues[0].field;
};

/**
 * Extract value at specific x-axis index from metrics
 * Generic: works with any x-axis field
 * @param {Array} metricsArray - Array of metric objects
 * @param {string} metricKey - The metric field name
 * @param {number} xValue - The x-axis value to find
 * @param {string} xAxisField - The x-axis field name
 * @returns {number|null} - Value at that x-axis point, or null if not found
 */
const getValueAtX = (metricsArray, metricKey, xValue, xAxisField) => {
  const metricAtX = metricsArray.find(m => m[xAxisField] === xValue);
  return metricAtX?.[metricKey] ?? null;
};

/**
 * Generic aggregation function for finding min/max/best metric values
 *
 * Scans through metrics array and finds the optimal value according to the specified mode.
 * Returns both the value and the x-axis coordinate where it occurred.
 *
 * Modes:
 * - 'min': Find minimum value (good for loss, error)
 * - 'max': Find maximum value (good for accuracy, F1)
 * - 'best': Auto-detect based on metric name (min for loss/error, max for others)
 *
 * @param {Array<object>} metricsArray - Array of metric records with x-axis and metric values
 * @param {string} metricKey - The metric field to aggregate (e.g., "loss", "accuracy")
 * @param {string} mode - Aggregation mode: 'min', 'max', or 'best'
 * @returns {{value: number, xValue: number, xField: string}|null} Result object:
 *   - value: The aggregated metric value
 *   - xValue: X-axis value where this occurred (e.g., epoch 42)
 *   - xField: Name of x-axis field (e.g., "epoch")
 *   Returns null if no valid numeric values found
 *
 * @example
 * const metrics = [
 *   { epoch: 0, loss: 0.5 },
 *   { epoch: 1, loss: 0.3 },
 *   { epoch: 2, loss: 0.2 }
 * ];
 * aggregateValue(metrics, 'loss', 'min');
 * // Returns: { value: 0.2, xValue: 2, xField: 'epoch' }
 */
const aggregateValue = (metricsArray, metricKey, mode) => {
  if (!metricsArray || metricsArray.length === 0) return null;

  const xAxisField = detectXAxisField(metricsArray);
  let resultValue = null;
  let resultXValue = null;

  // Determine comparison function based on mode
  let shouldUpdate;
  if (mode === 'min') {
    /**
     * Check if new value is less than current value.
     * @param {number} newVal - New value to compare
     * @param {number} currentVal - Current value
     * @returns {boolean} True if new value is less
     */
    shouldUpdate = (newVal, currentVal) => newVal < currentVal;
  } else if (mode === 'max') {
    /**
     * Check if new value is greater than current value.
     * @param {number} newVal - New value to compare
     * @param {number} currentVal - Current value
     * @returns {boolean} True if new value is greater
     */
    shouldUpdate = (newVal, currentVal) => newVal > currentVal;
  } else if (mode === 'best') {
    const isLossMetric = metricKey.includes('loss') || metricKey.includes('error');
    shouldUpdate = isLossMetric
      ? (newVal, currentVal) => newVal < currentVal
      : (newVal, currentVal) => newVal > currentVal;
  }

  for (const metric of metricsArray) {
    const value = metric[metricKey];
    if (typeof value !== 'number') continue;

    if (resultValue === null || shouldUpdate(value, resultValue)) {
      resultValue = value;
      resultXValue = metric[xAxisField];
    }
  }

  return resultValue !== null ? { value: resultValue, xValue: resultXValue, xField: xAxisField } : null;
};

/**
 * Get best value for a metric (min for loss/error, max for others)
 * Generic: returns x-axis value instead of hardcoded 'epoch'
 * @param {Array} metricsArray - Array of metric objects
 * @param {string} metricKey - The metric field name
 * @returns {{value: number, xValue: number, xField: string}|null} - Best value and its x-axis point
 */

/**
 * Get final (last) value for a metric
 * Generic: works with any data structure
 * Searches backwards to find the most recent occurrence
 * @param {Array} metricsArray - Array of metric objects
 * @param {string} metricKey - The metric field name
 * @returns {{value: number, xValue: number, xField: string}|null} - Final value and its x-axis point
 */
const getFinalValue = (metricsArray, metricKey) => {
  if (!metricsArray || metricsArray.length === 0) return null;

  const xAxisField = detectXAxisField(metricsArray);

  // Search backwards to find most recent value (metrics may be sparse)
  for (let i = metricsArray.length - 1; i >= 0; i--) {
    const metric = metricsArray[i];
    const value = metric?.[metricKey];

    if (typeof value === 'number') {
      return { value, xValue: metric[xAxisField], xField: xAxisField };
    }
  }

  return null;
};

/**
 * Discover all unique SCALAR metric keys from metrics array
 * Fully agnostic - only includes numeric scalar values
 * Excludes: internal fields (_*), arrays, objects
 * @param {Array} metrics - Array of metric objects
 * @returns {Array<string>} - Array of scalar metric keys
 */

/**
 * Discover all unique metric keys across multiple runs
 * Uses structured_data Series primitives
 * @param {Array} runs - Array of run objects
 * @returns {Array<string>} - Array of metric keys
 */

/**
 * Discover metrics organized by stream
 * Returns a map of streamId -> metric keys
 * @param {Array} runs - Array of run objects
 * @returns {object} - Object mapping streamId to array of metric keys
 */
export const discoverMetricsByStream = (runs) => {
  const seriesMetrics = {}; // seriesName -> Set of metric keys

  for (const run of runs) {
    if (!run.structured_data) continue;

    Object.entries(run.structured_data).forEach(([name, entries]) => {
      // Use most recent entry
      const latestEntry = entries[entries.length - 1];

      if (latestEntry.primitive_type !== 'series') return;

      const { data } = latestEntry;
      if (!data || !data.fields) return;

      // Initialize set for this series
      if (!seriesMetrics[name]) {
        seriesMetrics[name] = new Set();
      }

      // Add all field names (metrics) from this series
      Object.keys(data.fields).forEach(fieldName => {
        seriesMetrics[name].add(fieldName);
      });
    });
  }

  // Convert Sets to sorted arrays
  const result = {};
  Object.entries(seriesMetrics).forEach(([seriesName, keysSet]) => {
    result[seriesName] = Array.from(keysSet).sort();
  });

  return result;
};

/**
 * Discover Series primitives from structured_data (NEW - uses primitives)
 * @param {Array} runs - Array of run objects
 * @returns {object} - { seriesName: { metricKeys: [], runs: [{ run_id, data: [] }] } }
 */

/**
 * Get minimum value for a metric
 * Generic: works with any x-axis field
 * @param {Array} metricsArray - Array of metric objects
 * @param {string} metricKey - The metric field name
 * @returns {{value: number, xValue: number, xField: string}|null} - Min value and its x-axis point
 */
const getMinValue = (metricsArray, metricKey) => {
  return aggregateValue(metricsArray, metricKey, 'min');
};

/**
 * Get maximum value for a metric
 * Generic: works with any x-axis field
 * @param {Array} metricsArray - Array of metric objects
 * @param {string} metricKey - The metric field name
 * @returns {{value: number, xValue: number, xField: string}|null} - Max value and its x-axis point
 */
const getMaxValue = (metricsArray, metricKey) => {
  return aggregateValue(metricsArray, metricKey, 'max');
};

/**
 * Get metric value using specified aggregation strategy
 *
 * This is the main entry point for extracting a single aggregated metric value from a run.
 * Used extensively by the Tables tab to display summary statistics.
 *
 * The function searches through the run's structured_data for Series primitives containing
 * the requested metric, then applies the specified aggregation logic.
 *
 * Aggregation modes:
 * - 'min': Return metricKey value at the point where optimizeMetric is minimized
 *   Example: "Show accuracy when loss was lowest"
 * - 'max': Return metricKey value at the point where optimizeMetric is maximized
 *   Example: "Show loss when accuracy was highest"
 * - 'final': Return the last logged value of metricKey
 * - Numeric value: Return metricKey value at that specific x-axis point
 *
 * @param {object} run - Run object with structured_data containing Series primitives
 * @param {string} metricKey - The metric to extract (e.g., "accuracy", "val_loss")
 * @param {object} [aggregation={mode:'min',optimizeMetric:'loss'}] - Aggregation config:
 *   - mode: string - 'min', 'max', 'final', or numeric x-axis value
 *   - optimizeMetric: string - Metric to optimize for min/max modes
 *   - streamId: string (optional) - Specific Series primitive to search in
 * @returns {number|null} Aggregated metric value, or null if metric not found
 *
 * @example
 * // Get final validation loss
 * getMetricValue(run, 'val_loss', { mode: 'final' });
 *
 * // Get accuracy at the point where loss was minimized
 * getMetricValue(run, 'accuracy', { mode: 'min', optimizeMetric: 'loss' });
 *
 * // Get loss at epoch 50
 * getMetricValue(run, 'loss', { mode: 50 });
 */
export const getMetricValue = (run, metricKey, aggregation = { mode: 'min', optimizeMetric: 'loss' }) => {
  if (!run.structured_data) return null;

  const metricsArray = extractMetricsFromSeries(run.structured_data, metricKey, aggregation.streamId);
  if (!metricsArray) return null;

  return applyAggregation(metricsArray, metricKey, aggregation);
};

/**
 * Extract metrics array from Series primitives in structured_data.
 * @param {object} structured_data - Run's structured_data object
 * @param {string} metricKey - The metric field to find
 * @param {string} seriesId - Optional: specific series to look in
 * @returns {Array<object>|null} Array of metric objects or null
 */
function extractMetricsFromSeries(structured_data, metricKey, seriesId = null) {
  for (const [name, entries] of Object.entries(structured_data)) {
    // Use most recent entry
    const latestEntry = entries[entries.length - 1];

    if (latestEntry.primitive_type !== 'series') continue;

    // If seriesId specified, only look in that series
    if (seriesId && name !== seriesId) continue;

    const { data } = latestEntry;
    if (!data || !data.fields) continue;

    // Check if this series has the metric
    if (!data.fields[metricKey]) continue;

    // Series format: { index: 'epoch', fields: { loss: [0.5, 0.3], acc: [0.8, 0.9] }, index_values: [0, 1] }
    const indexValues = data.index_values || Array.from({ length: data.fields[metricKey].length }, (_, i) => i);
    const indexField = data.index || 'x';

    const metricsArray = indexValues.map((xValue, i) => {
      const record = { [indexField]: xValue };

      // Add all fields at this index
      Object.entries(data.fields).forEach(([fieldName, values]) => {
        record[fieldName] = values[i];
      });

      return record;
    });

    return metricsArray;
  }

  return null;
}

/**
 * Apply aggregation logic to metrics array.
 * @param {Array<object>} metricsArray - Array of metric objects
 * @param {string} metricKey - The metric field to display
 * @param {object} aggregation - Aggregation configuration
 * @returns {number|null} Aggregated value or null
 */
function applyAggregation(metricsArray, metricKey, aggregation) {
  const { mode, optimizeMetric } = aggregation;

  if (mode === 'min') {
    const streamHasOptimizeMetric = metricsArray.some(record => record[optimizeMetric] !== undefined);
    if (!streamHasOptimizeMetric) {
      const result = getFinalValue(metricsArray, metricKey);
      return result?.value ?? null;
    }

    // Find x-value where optimizeMetric is minimum, then get metricKey from that x-value
    const result = getMinValue(metricsArray, optimizeMetric);
    if (!result) return null;
    return getValueAtX(metricsArray, metricKey, result.xValue, result.xField);
  } else if (mode === 'max') {
    const streamHasOptimizeMetric = metricsArray.some(record => record[optimizeMetric] !== undefined);
    if (!streamHasOptimizeMetric) {
      const result = getFinalValue(metricsArray, metricKey);
      return result?.value ?? null;
    }

    // Find x-value where optimizeMetric is maximum, then get metricKey from that x-value
    const result = getMaxValue(metricsArray, optimizeMetric);
    if (!result) return null;
    return getValueAtX(metricsArray, metricKey, result.xValue, result.xField);
  } else if (mode === 'final') {
    const result = getFinalValue(metricsArray, metricKey);
    return result?.value ?? null;
  } else {
    // Specific x-axis value - need to detect x-field first
    const xAxisField = detectXAxisField(metricsArray);
    return getValueAtX(metricsArray, metricKey, mode, xAxisField);
  }
}

/**
 * Detect x-axis candidate fields using heuristics
 * These are fields likely used FOR plotting (not plotted themselves)
 * @param {Array} metricsArray - Array of metric objects
 * @returns {Set<string>} - Set of field names that are likely x-axis fields
 */

/**
 * Collect plottable metric keys from metrics array
 * Excludes internal fields, x-axis fields, and special structures
 * @param {Array} metricsArray - Array of metric objects
 * @param {Set} xAxisCandidates - Set of x-axis field names to exclude
 * @returns {Set<string>} - Set of plottable metric keys
 */

/**
 * Parse metric key into prefix and base metric name
 * @param {string} key - Metric key (e.g., 'train_loss', 'val_accuracy', 'cpu_percent')
 * @returns {{prefix: string|null, baseMetric: string}} - Parsed components
 */

/**
 * Convert snake_case to camelCase
 * @param {string} str - Snake case string
 * @returns {string} - Camel case string
 */

/**
 * Create history key for a metric
 * @param {string} metricKey - Original metric key
 * @returns {string} - History key for allData object
 */
