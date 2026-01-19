/**
 * Utility functions for cross-run comparison analysis
 * Supports parameter correlation and value extraction from runs
 */

/**
 * Get short display name for a parameter (last segment after period)
 * @param {string} paramName - Full parameter name (e.g., "training.optimizer.learningRate")
 * @returns {string} - Short name (e.g., "learningRate")
 */
export function getShortParamName(paramName) {
  if (!paramName) return '';
  const lastDot = paramName.lastIndexOf('.');
  return lastDot !== -1 ? paramName.slice(lastDot + 1) : paramName;
}

/**
 * Extract values for specified fields from a run
 * For metrics: aggregates values based on the specified method
 * @param {Object} run - The run object
 * @param {Array} fields - Fields to extract
 * @param {String} aggregation - 'last', 'max', 'min', or 'avg' (default: 'last')
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
          case 'avg':
            values[field] = fieldValues.reduce((a, b) => a + b, 0) / fieldValues.length;
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
 * Calculate parameter importance using correlation analysis
 * Returns importance scores for each hyperparameter relative to each metric
 *
 * Note: Full random forest implementation would require ML library
 * For now, using Pearson correlation as a simpler alternative
 */
export function calculateParameterImportance(runs, hyperparameters, metrics, aggregation = 'last') {
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
 * Calculate Pearson correlation coefficient
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
 * Convert values to numeric (handle categorical variables)
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
