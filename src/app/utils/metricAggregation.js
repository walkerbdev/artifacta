/**
 * Client-side metric aggregation utilities
 * FULLY GENERIC - no assumptions about data structure
 */

/**
 * Detect x-axis field from metrics array
 * Generic: finds integer fields that appear in all entries
 * @param {Array} metricsArray - Array of metric objects
 * @returns {string|null} - X-axis field name or null
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
 * Generic aggregation function - replaces getBestValue, getMinValue, getMaxValue
 * @param {Array} metricsArray - Array of metric objects
 * @param {string} metricKey - The metric field name
 * @param {string} mode - 'min', 'max', or 'best' (auto min/max based on metric name)
 * @returns {{value: number, xValue: number, xField: string}|null} - Aggregated value and its x-axis point
 */
const aggregateValue = (metricsArray, metricKey, mode) => {
  if (!metricsArray || metricsArray.length === 0) return null;

  const xAxisField = detectXAxisField(metricsArray);
  let resultValue = null;
  let resultXValue = null;

  // Determine comparison function based on mode
  let shouldUpdate;
  if (mode === 'min') {
    shouldUpdate = (newVal, currentVal) => newVal < currentVal;
  } else if (mode === 'max') {
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
 * @returns {Object} - Object mapping streamId to array of metric keys
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
 * @returns {Object} - { seriesName: { metricKeys: [], runs: [{ run_id, data: [] }] } }
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
 * Get value based on aggregation mode
 * Searches structured_data Series primitives for the metric
 * @param {Object} run - Run object
 * @param {string} metricKey - The metric field to display
 * @param {Object} aggregation - Aggregation configuration
 * @param {string} aggregation.mode - 'min', 'max', 'final', or x-axis value
 * @param {string} aggregation.optimizeMetric - Metric to optimize (for min/max modes)
 * @param {string} aggregation.streamId - Optional: specific series to look in
 * @returns {number|null} - Value at the specified mode
 */
export const getMetricValue = (run, metricKey, aggregation = { mode: 'min', optimizeMetric: 'loss' }) => {
  if (!run.structured_data) return null;

  const metricsArray = extractMetricsFromSeries(run.structured_data, metricKey, aggregation.streamId);
  if (!metricsArray) return null;

  return applyAggregation(metricsArray, metricKey, aggregation);
};

/**
 * Extract metrics array from Series primitives in structured_data
 * @param {Object} structured_data - Run's structured_data object
 * @param {string} metricKey - The metric field to find
 * @param {string} seriesId - Optional: specific series to look in
 * @returns {Array|null} - Array of metric objects [{x, metricKey}, ...]
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
 * Apply aggregation logic to metrics array
 * @param {Array} metricsArray - Array of metric objects
 * @param {string} metricKey - The metric field to display
 * @param {Object} aggregation - Aggregation configuration
 * @returns {number|null} - Aggregated value
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
