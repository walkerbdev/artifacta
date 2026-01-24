/**
 * Utility functions for cross-run comparison analysis
 *
 * This module provides tools for analyzing relationships between hyperparameters
 * and metrics across multiple experiment runs. The primary use case is hyperparameter
 * sweep analysis, where we want to understand which parameters most strongly affect
 * which metrics.
 *
 * Key capabilities:
 * - Extract parameter/metric values from runs with flexible aggregation
 * - Calculate Pearson correlation between parameters and metrics
 * - Handle both numeric and categorical parameters
 * - Support multiple aggregation strategies (last, max, min, avg)
 */

/**
 * Get short display name for a parameter by extracting the last segment after a period
 *
 * Useful for displaying nested parameter names in charts where space is limited.
 * For example, "model.optimizer.learning_rate" becomes "learning_rate".
 *
 * @param {string} paramName - Full parameter name (e.g., "training.optimizer.learningRate")
 * @returns {string} Short name (e.g., "learningRate"), or empty string if input is falsy
 *
 * @example
 * getShortParamName("model.optimizer.lr") // Returns "lr"
 * getShortParamName("batch_size")         // Returns "batch_size" (no dots)
 * getShortParamName("")                   // Returns ""
 */
export function getShortParamName(paramName) {
  if (!paramName) return '';
  const lastDot = paramName.lastIndexOf('.');
  return lastDot !== -1 ? paramName.slice(lastDot + 1) : paramName;
}

/**
 * Extract values for specified fields from a run, with flexible aggregation for metrics
 *
 * This function searches for field values in multiple locations within a run object:
 * 1. run.params - Single-value parameters (no aggregation)
 * 2. run.config - Configuration values (no aggregation)
 * 3. run.tags - Tag values (no aggregation)
 * 4. run.structured_data - Series data that requires aggregation
 *
 * For series data (metrics logged over time), the aggregation parameter controls
 * how multiple values are reduced to a single value for comparison.
 *
 * @param {object} run - The run object containing params, config, tags, and structured_data
 * @param {Array<string>} fields - Field names to extract (e.g., ["learning_rate", "train_loss"])
 * @param {string} [aggregation='last'] - How to aggregate series data:
 *   - 'last': Use final value (default, good for final metrics)
 *   - 'max': Use maximum value (good for accuracy, best performance)
 *   - 'min': Use minimum value (good for loss, error rates)
 *   - 'avg': Use average value (good for stability analysis)
 * @returns {object} Object mapping field names to extracted values. Missing fields map to null.
 *
 * @example
 * const run = {
 *   config: { learning_rate: 0.01 },
 *   structured_data: {
 *     metrics: [{
 *       primitive_type: 'series',
 *       data: { fields: { loss: [0.5, 0.3, 0.2] } }
 *     }]
 *   }
 * };
 * extractValues(run, ['learning_rate', 'loss'], 'last');
 * // Returns { learning_rate: 0.01, loss: 0.2 }
 */
export function extractValues(run, fields, aggregation = 'last') {
  const values = {};

  fields.forEach(field => {
    // Check params, config, and tags first (these are single values, no aggregation needed)
    if (run.params && field in run.params) {
      values[field] = run.params[field];
      return;
    }
    if (run.config && field in run.config) {
      values[field] = run.config[field];
      return;
    }
    if (run.tags && field in run.tags) {
      values[field] = run.tags[field];
      return;
    }

    // Check structured_data for metrics
    if (run.structured_data) {
      const fieldValues = [];

      Object.entries(run.structured_data).forEach(([_name, entries]) => {
        if (!entries || entries.length === 0) return;

        const latestEntry = entries[entries.length - 1];
        const data = latestEntry.data;

        // Extract from series fields
        if (latestEntry.primitive_type === 'series' && data.fields && data.fields[field]) {
          const values = data.fields[field];
          if (Array.isArray(values) && values.length > 0) {
            fieldValues.push(...values);
          }
        }
      });

      if (fieldValues.length > 0) {
        switch (aggregation) {
          case 'max':
            values[field] = Math.max(...fieldValues);
            break;
          case 'min':
            values[field] = Math.min(...fieldValues);
            break;
          case 'last':
          default:
            values[field] = fieldValues[fieldValues.length - 1];
            break;
        }
        return;
      }
    }

    // Field not found - set to null
    values[field] = null;
  });

  return values;
}

/**
 * Calculate parameter correlation using Pearson correlation analysis
 *
 * Analyzes the linear relationship between hyperparameters and metrics across multiple runs.
 * This is useful for understanding which hyperparameters most strongly affect which metrics
 * in a hyperparameter sweep.
 *
 * Algorithm:
 * 1. For each metric and each hyperparameter pair:
 *    - Extract values from all runs using specified aggregation
 *    - Filter out runs where either value is missing
 *    - Calculate Pearson correlation coefficient (-1 to +1)
 *    - Store both correlation and absolute value (importance score)
 *
 * 2. Pearson correlation measures linear relationship:
 *    - +1.0: Perfect positive correlation (param ↑ → metric ↑)
 *    - 0.0: No linear correlation
 *    - -1.0: Perfect negative correlation (param ↑ → metric ↓)
 *
 * 3. Importance score is |correlation|, treating strong negative correlations
 *    as equally important to strong positive ones.
 *
 * Limitations:
 * - Only detects LINEAR relationships (won't catch quadratic, exponential, etc.)
 * - Correlation ≠ causation
 * - Not true feature importance (use permutation importance, SHAP, or random forests for that)
 * - Requires at least 3 runs for meaningful results
 * - Categorical parameters are converted to ordinal (0, 1, 2...) which may be misleading
 *
 * @param {Array<object>} runs - Array of run objects with params/config/structured_data
 * @param {Array<string>} hyperparameters - Parameter names to analyze (e.g., ["learning_rate", "batch_size"])
 * @param {Array<string>} metrics - Metric names to analyze (e.g., ["accuracy", "loss"])
 * @param {string} [aggregation='last'] - How to aggregate metric series ('last', 'max', 'min')
 * @returns {object|null} Nested object: { metric: { param: { correlation, importance } } }
 *   Returns null if fewer than 3 runs (insufficient for correlation)
 *
 * @example
 * const runs = [
 *   { config: { lr: 0.01 }, structured_data: {...} },  // final loss: 0.5
 *   { config: { lr: 0.1 },  structured_data: {...} },  // final loss: 0.3
 *   { config: { lr: 1.0 },  structured_data: {...} }   // final loss: 0.8
 * ];
 * const result = calculateParameterCorrelation(runs, ['lr'], ['loss'], 'last');
 * // Returns: { loss: { lr: { correlation: 0.5, importance: 0.5 } } }
 */
export function calculateParameterCorrelation(runs, hyperparameters, metrics, aggregation = 'last') {
  if (runs.length < 3) {
    // Need at least 3 runs for meaningful correlation
    return null;
  }

  const results = {};

  metrics.forEach(metric => {
    results[metric] = {};

    hyperparameters.forEach(param => {
      // Extract param and metric values for all runs with specified aggregation
      const pairs = runs.map(run => {
        const values = extractValues(run, [param, metric], aggregation);
        return { param: values[param], metric: values[metric] };
      }).filter(pair => pair.param !== null && pair.metric !== null);

      if (pairs.length < 3) {
        results[metric][param] = { correlation: 0, importance: 0 };
        return;
      }

      // Calculate Pearson correlation
      const correlation = calculateCorrelation(
        pairs.map(p => p.param),
        pairs.map(p => p.metric)
      );

      // Use absolute correlation as importance score
      // (both positive and negative correlations are "important")
      const importance = Math.abs(correlation);

      results[metric][param] = { correlation, importance };
    });
  });

  return results;
}

/**
 * Calculate Pearson correlation coefficient between two arrays
 *
 * Pearson correlation formula: r = Σ[(xi - x̄)(yi - ȳ)] / sqrt(Σ(xi - x̄)² * Σ(yi - ȳ)²)
 * Simplified to: r = [n*Σ(xy) - Σx*Σy] / sqrt([n*Σx² - (Σx)²] * [n*Σy² - (Σy)²])
 *
 * The coefficient ranges from -1 to +1:
 * - r = +1: Perfect positive linear relationship
 * - r = 0: No linear relationship
 * - r = -1: Perfect negative linear relationship
 *
 * Handles categorical data by converting to ordinal indices (0, 1, 2, ...).
 *
 * @param {Array<number>} x - First array of values
 * @param {Array<number>} y - Second array of values (must be same length as x)
 * @returns {number} Correlation coefficient between -1 and 1, or 0 if calculation fails
 *   (e.g., arrays different lengths, < 2 values, zero variance)
 */
function calculateCorrelation(x, y) {
  const n = x.length;
  if (n !== y.length || n < 2) return 0;

  // Convert categorical to numeric if needed
  const xNumeric = convertToNumeric(x);
  const yNumeric = convertToNumeric(y);

  const sumX = xNumeric.reduce((a, b) => a + b, 0);
  const sumY = yNumeric.reduce((a, b) => a + b, 0);
  const sumXY = xNumeric.reduce((sum, xi, i) => sum + xi * yNumeric[i], 0);
  const sumX2 = xNumeric.reduce((sum, xi) => sum + xi * xi, 0);
  const sumY2 = yNumeric.reduce((sum, yi) => sum + yi * yi, 0);

  const numerator = n * sumXY - sumX * sumY;
  const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

  if (denominator === 0) return 0;

  return numerator / denominator;
}

/**
 * Convert values to numeric, handling categorical variables
 *
 * For numeric arrays: returns as-is
 * For categorical arrays: maps unique values to ordinal indices
 *
 * Example: ["adam", "sgd", "adam", "rmsprop"] → [0, 1, 0, 2]
 *
 * Warning: This treats categorical values as ordinal, which may not be
 * semantically correct (e.g., "adam" isn't "less than" "sgd"). Consider
 * one-hot encoding for proper categorical analysis.
 *
 * @param {Array<unknown>} values - Array of values to convert (can be numbers, strings, etc.)
 * @returns {Array<number>} Array of numeric values. Categorical values mapped to indices (0, 1, 2, ...)
 */
function convertToNumeric(values) {
  // Already numeric
  if (values.every(v => typeof v === 'number')) {
    return values;
  }

  // Categorical - assign indices
  const uniqueValues = [...new Set(values)];
  const valueToIndex = new Map(uniqueValues.map((v, i) => [v, i]));

  return values.map(v => valueToIndex.get(v));
}
