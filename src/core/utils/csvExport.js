/**
 * CSV Export Utilities
 * Shared functions for exporting data to CSV files
 */

/**
 * Escapes a field for CSV format.
 * @param {string|number|boolean} field - The field to escape
 * @returns {string} Escaped field
 */
function escapeCSV(field) {
  const str = String(field);
  if (str.includes(',') || str.includes('"') || str.includes('\n')) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

/**
 * Formats a value for CSV export.
 * @param {string|number|null|undefined} value - The value to format
 * @param {number} decimals - Number of decimal places for numbers (default: 4)
 * @returns {string} Formatted value
 */
function formatCSVValue(value, decimals = 4) {
  if (value === null || value === undefined) return '-';
  if (typeof value === 'number') return value.toFixed(decimals);
  return String(value);
}

/**
 * Downloads data as a CSV file
 * @param {Array<Array<string>>} rows - 2D array of data (including header row)
 * @param {string} filename - Name of the file to download
 */
function downloadCSV(rows, filename) {
  // Convert to CSV string
  const csvContent = rows
    .map(row => row.map(escapeCSV).join(','))
    .join('\n');

  // Download file
  const blob = new Blob([csvContent], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Helper to create a timestamped filename
 * @param {string} prefix - Filename prefix
 * @returns {string} Filename with timestamp
 */
function createTimestampedFilename(prefix) {
  const date = new Date().toISOString().slice(0, 10);
  return `${prefix}_${date}.csv`;
}

/**
 * Convert snake_case or kebab-case to Title Case for display
 * @param {string} key - The key to convert
 * @returns {string} Title cased string
 */
function getDisplayName(key) {
  return key.split('_').map(word =>
    word.charAt(0).toUpperCase() + word.slice(1)
  ).join(' ');
}

/**
 * Groups series by their metric signature
 * Series with the same set of metrics are grouped together.
 * @param {object} metricsByStream - Object mapping stream names to metric keys
 * @returns {object} Series groups indexed by signature
 */
function groupSeriesBySignature(metricsByStream) {
  const seriesGroups = {};
  Object.entries(metricsByStream).forEach(([seriesName, metricKeys]) => {
    const signature = metricKeys.slice().sort().join(',');
    if (!seriesGroups[signature]) {
      seriesGroups[signature] = [];
    }
    seriesGroups[signature].push({ seriesName, metricKeys });
  });
  return seriesGroups;
}

/**
 * Filters runs to only those containing series data
 * @param {Array} runs - Array of run objects
 * @returns {Array} Filtered runs with series data
 */
function filterRunsWithSeries(runs) {
  return runs.filter(run => {
    return run.structured_data && Object.values(run.structured_data).some(entries => {
      const latestEntry = entries[entries.length - 1];
      return latestEntry.primitive_type === 'series';
    });
  });
}

/**
 * Builds CSV rows for a specific series group.
 * @param {Array<object>} runs - Runs to include in this group
 * @param {Array<string>} metricKeys - Metric keys for this group
 * @param {Array<object>} seriesGroup - Series information for this group
 * @param {(run: object, key: string, options: object) => string|number} getMetricValue - Function to extract metric values
 * @param {object} aggSettings - Aggregation settings (mode, optimizeMetric)
 * @returns {Array<Array<string>>} 2D array of CSV rows (header + data)
 */
function buildSeriesGroupCSV(runs, metricKeys, seriesGroup, getMetricValue, aggSettings) {
  const { mode: aggregationMode, optimizeMetric } = aggSettings;
  const rows = [];

  // Header row
  const headerRow = ['Run ID'];
  metricKeys.forEach(key => headerRow.push(getDisplayName(key)));
  rows.push(headerRow);

  // Data rows
  runs.forEach(run => {
    const runSeriesName = seriesGroup.find(({ seriesName }) => {
      return run.structured_data?.[seriesName];
    })?.seriesName;

    const row = [run.name || run.run_id.substring(0, 20)];
    metricKeys.forEach(key => {
      const value = getMetricValue(run, key, { mode: aggregationMode, optimizeMetric, streamId: runSeriesName });
      row.push(formatCSVValue(value));
    });
    rows.push(row);
  });

  return rows;
}

/**
 * Generates filename for a series group
 * @param {Array} seriesGroup - Series information for this group
 * @returns {string} Base filename for this group
 */
function getSeriesGroupFilename(seriesGroup) {
  return seriesGroup.length === 1
    ? seriesGroup[0].seriesName
    : seriesGroup.map(s => s.seriesName).join('_');
}

/**
 * Exports all series groups as separate CSV files
 * Main orchestration function that handles the complete export workflow.
 * @param {Array<object>} runs - All runs to export
 * @param {object} metricsByStream - Metrics organized by stream
 * @param {object} streamAggregation - Aggregation settings per group
 * @param {(run: object, key: string, options: object) => string|number} getMetricValue - Function to extract metric values
 * @returns {void}
 */
export function exportSeriesGroupsAsCSV(runs, metricsByStream, streamAggregation, getMetricValue) {
  if (runs.length === 0) return;

  // Group series by metric signature
  const seriesGroups = groupSeriesBySignature(metricsByStream);

  // Filter to runs with series data
  const runsWithSeries = filterRunsWithSeries(runs);

  // Export each group as a separate CSV file
  Object.entries(seriesGroups).forEach(([_signature, seriesGroup], groupIdx) => {
    const metricKeys = seriesGroup[0].metricKeys;
    const groupKey = `group-${groupIdx}`;
    const aggSettings = streamAggregation[groupKey] || {
      mode: 'final',
      optimizeMetric: metricKeys[0] || 'loss'
    };
    const { mode: aggregationMode, optimizeMetric } = aggSettings;

    // Filter runs for this group
    const groupRuns = runsWithSeries.filter(run => {
      return seriesGroup.some(({ seriesName }) => {
        return metricKeys.some(key => {
          const value = getMetricValue(run, key, { mode: aggregationMode, optimizeMetric, streamId: seriesName });
          return value !== null && value !== undefined;
        });
      });
    });

    if (groupRuns.length === 0) return;

    // Build and download CSV
    const rows = buildSeriesGroupCSV(groupRuns, metricKeys, seriesGroup, getMetricValue, aggSettings);
    const filename = getSeriesGroupFilename(seriesGroup);
    downloadCSV(rows, createTimestampedFilename(filename));
  });
}
