import React, { useState, useMemo } from 'react';
import './RunFilter.scss';
import { getMetricValue } from '@/app/utils/metricAggregation';

/**
 * Run Filter component for dynamic filtering of experiment runs
 *
 * W&B-inspired filtering system that auto-discovers filter options from run data.
 * No hardcoded metric names - adapts to whatever metrics users log.
 *
 * Features:
 * - Text search (run name, run ID)
 * - Status filtering (Running, Completed, Failed)
 * - Metric threshold filtering (min/max ranges per metric)
 * - Auto-discovery of available metrics across all runs
 * - Collapsible filter panel
 * - Real-time filtering (updates as user types)
 * - Respects aggregation mode for metric values
 *
 * Filter types:
 * 1. Search: Fuzzy match on run name or ID
 * 2. Status: Running (in progress) vs Completed/Failed
 * 3. Metrics: Min/max range filters for any logged metric
 *
 * Architecture:
 * - Stateless filtering (pure function applied to runs)
 * - Callback-based (notifies parent of filtered results)
 * - Metric discovery via structured_data inspection
 *
 * @param {object} props - Component props
 * @param {Array<object>} [props.runs=[]] - All available runs to filter
 * @param {function} props.onFilterChange - Callback with filtered results
 *   Signature: (filteredRuns: Array<object>) => void
 * @param {string} [props.aggregationMode='min'] - How to aggregate metrics ('min'/'max'/'final')
 * @param {string} [props.optimizeMetric='loss'] - Metric to optimize for min/max modes
 * @returns {React.ReactElement} Collapsible filter panel
 */
export const RunFilter = ({ runs = [], onFilterChange, aggregationMode = 'min', optimizeMetric = 'loss' }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('all'); // 'all', 'running', 'completed'
  const [metricFilters, setMetricFilters] = useState({}); // { metric_name: { min: val, max: val } }
  const [showFilters, setShowFilters] = useState(false);

  // Auto-discover all available metrics across all runs
  const availableMetrics = useMemo(() => {
    const metricSet = new Set();
    runs.forEach(run => {
      if (run.structured_data) {
        // Loop over all series in structured_data
        Object.values(run.structured_data).forEach(entries => {
          if (!entries || entries.length === 0) return;

          const latestEntry = entries[entries.length - 1];
          if (latestEntry.primitive_type !== 'series') return;

          const { data } = latestEntry;
          if (!data || !data.fields) return;

          // Add all field names from this series
          Object.keys(data.fields).forEach(key => {
            if (!key.startsWith('_')) {
              metricSet.add(key);
            }
          });
        });
      }
    });
    return Array.from(metricSet).sort();
  }, [runs]);

  // Auto-discover available statuses
  const availableStatuses = useMemo(() => {
    const statuses = new Set();
    runs.forEach(run => {
      if (run.status) statuses.add(run.status);
    });
    return Array.from(statuses);
  }, [runs]);

  /**
   * Format metric name for display
   * @param {string} metricKey - The metric key to format
   * @returns {string} The formatted metric name
   */
  const formatMetricName = (metricKey) => {
    return metricKey
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  // Apply all filters
  const filteredRuns = useMemo(() => {
    let filtered = runs;

    // Search filter (name or run_id)
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(run =>
        run.name?.toLowerCase().includes(query) ||
        run.run_id?.toLowerCase().includes(query)
      );
    }

    // Status filter
    if (statusFilter !== 'all') {
      filtered = filtered.filter(run => {
        if (statusFilter === 'running') return run.status === 'Running';
        if (statusFilter === 'completed') return run.status === 'Completed';
        return true;
      });
    }

    // Metric threshold filters
    const aggregation = { mode: aggregationMode, optimizeMetric };
    Object.entries(metricFilters).forEach(([metricKey, thresholds]) => {
      if (thresholds.min !== undefined && thresholds.min !== '') {
        filtered = filtered.filter(run => {
          const value = getMetricValue(run, metricKey, aggregation);
          return value !== null && value !== undefined && value >= parseFloat(thresholds.min);
        });
      }
      if (thresholds.max !== undefined && thresholds.max !== '') {
        filtered = filtered.filter(run => {
          const value = getMetricValue(run, metricKey, aggregation);
          return value !== null && value !== undefined && value <= parseFloat(thresholds.max);
        });
      }
    });

    return filtered;
  }, [runs, searchQuery, statusFilter, metricFilters, aggregationMode, optimizeMetric]);

  // Notify parent of filtered results
  React.useEffect(() => {
    onFilterChange(filteredRuns);
  }, [filteredRuns, onFilterChange]);

  /**
   * Handle metric filter change
   * @param {string} metricKey - The metric key to filter
   * @param {string} type - The type of threshold (min or max)
   * @param {string} value - The threshold value
   * @returns {void}
   */
  const handleMetricFilterChange = (metricKey, type, value) => {
    setMetricFilters(prev => ({
      ...prev,
      [metricKey]: {
        ...prev[metricKey],
        [type]: value
      }
    }));
  };

  /**
   * Clear a specific metric filter
   * @param {string} metricKey - The metric key to clear
   * @returns {void}
   */
  const clearMetricFilter = (metricKey) => {
    setMetricFilters(prev => {
      const newFilters = { ...prev };
      delete newFilters[metricKey];
      return newFilters;
    });
  };

  /**
   * Clear all filters and reset to default state
   * @returns {void}
   */
  const clearAllFilters = () => {
    setSearchQuery('');
    setStatusFilter('all');
    setMetricFilters({});
  };

  const hasActiveFilters = searchQuery || statusFilter !== 'all' || Object.keys(metricFilters).length > 0;

  return (
    <div className="run-filter">
      <div className="filter-header">
        <div className="filter-search">
          <input
            type="text"
            placeholder="Search by run name or ID..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="search-input"
          />
        </div>

        <button
          className="toggle-filters-btn"
          onClick={() => setShowFilters(!showFilters)}
        >
          {showFilters ? '▼' : '▶'} Filters
          {hasActiveFilters && <span className="active-filter-count">({Object.keys(metricFilters).length + (statusFilter !== 'all' ? 1 : 0)})</span>}
        </button>

        {hasActiveFilters && (
          <button
            className="clear-filters-btn"
            onClick={clearAllFilters}
            title="Clear all filters"
          >
            Clear All
          </button>
        )}
      </div>

      {showFilters && (
        <div className="filter-panel">
          {/* Status Filter */}
          {availableStatuses.length > 0 && (
            <div className="filter-section">
              <label className="filter-label">Status</label>
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="filter-select"
              >
                <option value="all">All ({runs.length})</option>
                {availableStatuses.includes('Running') && (
                  <option value="running">
                    Running ({runs.filter(r => r.status === 'Running').length})
                  </option>
                )}
                {availableStatuses.includes('Completed') && (
                  <option value="completed">
                    Completed ({runs.filter(r => r.status === 'Completed').length})
                  </option>
                )}
              </select>
            </div>
          )}

          {/* Metric Threshold Filters */}
          <div className="filter-section">
            <label className="filter-label">Metric Thresholds</label>

            {/* Add Metric Filter Dropdown */}
            <select
              className="add-metric-filter"
              onChange={(e) => {
                if (e.target.value) {
                  handleMetricFilterChange(e.target.value, 'min', '');
                  e.target.value = '';
                }
              }}
            >
              <option value="">+ Add metric filter...</option>
              {availableMetrics
                .filter(metric => !metricFilters[metric])
                .map(metric => (
                  <option key={metric} value={metric}>
                    {formatMetricName(metric)}
                  </option>
                ))}
            </select>

            {/* Active Metric Filters */}
            {Object.keys(metricFilters).length > 0 && (
              <div className="active-metric-filters">
                {Object.keys(metricFilters).map(metricKey => (
                  <div key={metricKey} className="metric-filter-row">
                    <div className="metric-filter-name">
                      {formatMetricName(metricKey)}
                    </div>
                    <div className="metric-filter-inputs">
                      <input
                        type="number"
                        placeholder="Min"
                        step="any"
                        value={metricFilters[metricKey]?.min ?? ''}
                        onChange={(e) => handleMetricFilterChange(metricKey, 'min', e.target.value)}
                        className="threshold-input"
                      />
                      <span>to</span>
                      <input
                        type="number"
                        placeholder="Max"
                        step="any"
                        value={metricFilters[metricKey]?.max ?? ''}
                        onChange={(e) => handleMetricFilterChange(metricKey, 'max', e.target.value)}
                        className="threshold-input"
                      />
                      <button
                        className="remove-filter-btn"
                        onClick={() => clearMetricFilter(metricKey)}
                        title="Remove filter"
                      >
                        ×
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
