import React, { useState } from 'react';
import { RunFilter } from './RunFilter';
import { CollapsibleButton } from './CollapsibleButton';
import { exportSeriesGroupsAsCSV } from '@/core/utils/csvExport';
import { discoverMetricsByStream, getMetricValue } from '@/app/utils/metricAggregation';
import './RunSelector.scss';

// Memoized config display to prevent flickering on re-renders
const ConfigDisplay = React.memo(({ config }) => (
  <pre className="config-json">{JSON.stringify(config, null, 2)}</pre>
), (prevProps, nextProps) => {
  // Only re-render if config content actually changed
  return JSON.stringify(prevProps.config) === JSON.stringify(nextProps.config);
});

/**
 * Run Selector Component
 *
 * Displays and manages run selection from the database.
 * Supports filtering, sorting, and multi-selection of runs.
 */
export const RunSelector = ({
  selectedRunIds = [], // NEW: controlled from parent
  onRunSelectionChange = () => {}, // NEW: callback to parent
  runs = [], // Runs data passed from parent (single source of truth)
  runsLoading = false // Loading state passed from parent
}) => {
  // No longer calling useRunData here - using props from parent to avoid duplicate polling
  const [expandedRunIds, setExpandedRunIds] = useState(new Set());
  // Per-stream aggregation settings: streamId -> { mode, optimizeMetric }
  const [streamAggregation, setStreamAggregation] = useState({});
  // Expanded tables: Set of table group keys (including 'metadata' for Run Metadata table)
  // Start with empty Set = all tables collapsed by default
  const [expandedTables, setExpandedTables] = useState(new Set());
  // Per-table filtered runs: tableKey -> filtered runs array
  const [tableFilteredRuns, setTableFilteredRuns] = useState({});

  // Export all visible tables as CSV
  const handleExportAllTables = () => {
    const selectedRuns = runs.filter(r => selectedRunIds.includes(r.run_id));
    const metricsByStream = discoverMetricsByStream(selectedRuns);
    exportSeriesGroupsAsCSV(selectedRuns, metricsByStream, streamAggregation, getMetricValue);
  };

  const toggleRunSelection = (runId) => {
    const newSelection = selectedRunIds.includes(runId)
      ? selectedRunIds.filter(id => id !== runId)
      : [...selectedRunIds, runId];
    onRunSelectionChange(newSelection);
  };

  const toggleExpandRun = (runId) => {
    setExpandedRunIds(prev => {
      const newSet = new Set(prev);
      if (newSet.has(runId)) {
        newSet.delete(runId);
      } else {
        newSet.add(runId);
      }
      return newSet;
    });
  };

  const toggleTableExpand = (tableKey) => {
    setExpandedTables(prev => {
      const newSet = new Set(prev);
      if (newSet.has(tableKey)) {
        newSet.delete(tableKey);
      } else {
        newSet.add(tableKey);
      }
      return newSet;
    });
  };

  // Always show the component - simplified to always show all runs
  return (
    <div className="run-selector">
      <div className="run-selector-header">
        <button
          onClick={handleExportAllTables}
          className="export-button"
          disabled={selectedRunIds.length === 0}
        >
          Export All Tables
        </button>
      </div>

      {/* Always show all runs view */}
      <div className="run-history-table">
        {runsLoading ? (
          <div className="loading">Loading runs...</div>
        ) : runs.length === 0 ? null : (() => {
          // Filter to only selected runs for table display
          const selectedRuns = runs.filter(r => selectedRunIds.includes(r.run_id));

          // If no runs selected, show nothing
          if (selectedRuns.length === 0) {
            return null;
          }

          // Discover metrics organized by stream (only from selected runs)
          const metricsByStream = discoverMetricsByStream(selectedRuns);

          // Discover all unique param/tag keys (shared across all tables)
          const allParamKeys = new Set();
          const allTagKeys = new Set();
          selectedRuns.forEach(run => {
            if (run.params) Object.keys(run.params).forEach(k => allParamKeys.add(k));
            if (run.tags) Object.keys(run.tags).forEach(k => allTagKeys.add(k));
          });

            // Show ALL params and tags - fully agnostic, no hardcoding
            const visibleParamKeys = Array.from(allParamKeys).sort();
            const visibleTagKeys = Array.from(allTagKeys).sort();

            const formatMetricValue = (key, value) => {
              if (value === null || value === undefined) return '-';
              return typeof value === 'number' ? value.toFixed(4) : String(value);
            };

            const getDisplayName = (key) => {
              return key.split('_').map(word =>
                word.charAt(0).toUpperCase() + word.slice(1)
              ).join(' ');
            };

            return (
              <>
                {/* Run Metadata Table (params and tags) */}
                {(visibleParamKeys.length > 0 || visibleTagKeys.length > 0) && (() => {
                  const isCollapsed = !expandedTables.has('metadata');
                  const tableKey = 'metadata';
                  const filteredMetadataRuns = tableFilteredRuns[tableKey] || selectedRuns;

                  return (
                    <div style={{ marginBottom: '24px', marginTop: '12px', position: 'relative' }}>
                      <h4 style={{ margin: 0, marginBottom: '12px', fontSize: '14px', fontWeight: '600', paddingRight: '80px' }}>
                        Run Metadata
                      </h4>
                      <CollapsibleButton
                        isCollapsed={isCollapsed}
                        onClick={() => toggleTableExpand(tableKey)}
                      />

                      {!isCollapsed && (
                        <>
                          <RunFilter
                            runs={selectedRuns}
                            onFilterChange={(filtered) => setTableFilteredRuns({ ...tableFilteredRuns, [tableKey]: filtered })}
                          />
                          <table>
                      <thead>
                        <tr>
                          <th style={{width: '30px'}}></th>
                          <th>
                            <input
                              type="checkbox"
                              checked={selectedRunIds.length === filteredMetadataRuns.length && filteredMetadataRuns.length > 0}
                              onChange={(e) => {
                                if (e.target.checked) {
                                  onRunSelectionChange(filteredMetadataRuns.map(r => r.run_id));
                                } else {
                                  onRunSelectionChange([]);
                                }
                              }}
                            />
                          </th>
                          <th>Run ID</th>
                          {visibleParamKeys.map(key => (
                            <th key={`param-${key}`} title="Parameter">{getDisplayName(key)}</th>
                          ))}
                          {visibleTagKeys.map(key => (
                            <th key={`tag-${key}`} title="Tag">{getDisplayName(key)}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {filteredMetadataRuns.map(run => (
                          <React.Fragment key={run.run_id}>
                            <tr
                              className={selectedRunIds.includes(run.run_id) ? 'selected' : ''}
                              onClick={() => toggleRunSelection(run.run_id)}
                            >
                              <td onClick={(e) => e.stopPropagation()}>
                                <button
                                  className="expand-btn"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    toggleExpandRun(run.run_id);
                                  }}
                                  disabled={!run.config}
                                >
                                  {run.config ? (expandedRunIds.has(run.run_id) ? '▼' : '▶') : '-'}
                                </button>
                              </td>
                              <td onClick={(e) => e.stopPropagation()}>
                                <input
                                  type="checkbox"
                                  checked={selectedRunIds.includes(run.run_id)}
                                  onChange={() => toggleRunSelection(run.run_id)}
                                />
                              </td>
                              <td title={run.run_id}>{run.name || run.run_id.substring(0, 20) + '...'}</td>
                              {visibleParamKeys.map(key => (
                                <td key={`param-${key}`} title={run.params?.[key] || '-'}>
                                  {run.params?.[key] ? String(run.params[key]).substring(0, 30) + (String(run.params[key]).length > 30 ? '...' : '') : '-'}
                                </td>
                              ))}
                              {visibleTagKeys.map(key => (
                                <td key={`tag-${key}`} title={run.tags?.[key] || '-'}>
                                  {run.tags?.[key] ? String(run.tags[key]).substring(0, 30) + (String(run.tags[key]).length > 30 ? '...' : '') : '-'}
                                </td>
                              ))}
                            </tr>
                            {expandedRunIds.has(run.run_id) && run.config && (
                              <tr className="config-row">
                                <td colSpan={3 + visibleParamKeys.length + visibleTagKeys.length}>
                                  <ConfigDisplay config={run.config} />
                                </td>
                              </tr>
                            )}
                          </React.Fragment>
                        ))}
                      </tbody>
                    </table>
                        </>
                      )}
                    </div>
                  );
                })()}

                {/* Group Series by metric signature - series with same metrics go in same table */}
                {Object.keys(metricsByStream).length > 0 && (() => {
                  // Group series by their metric signature (sorted metric keys)
                  const seriesGroups = {}; // signature -> [{ seriesName, metricKeys }]

                  Object.entries(metricsByStream).forEach(([seriesName, metricKeys]) => {
                    const signature = metricKeys.slice().sort().join(',');
                    if (!seriesGroups[signature]) {
                      seriesGroups[signature] = [];
                    }
                    seriesGroups[signature].push({ seriesName, metricKeys });
                  });

                  // Filter selected runs to only show those with Series data
                  const runsWithSeries = selectedRuns.filter(run => {
                    return run.structured_data && Object.values(run.structured_data).some(entries => {
                      const latestEntry = entries[entries.length - 1];
                      return latestEntry.primitive_type === 'series';
                    });
                  });

                  if (runsWithSeries.length === 0) return null;

                  return (
                    <>
                      {Object.entries(seriesGroups).map(([_signature, seriesGroup], groupIdx) => {
                        const metricKeys = seriesGroup[0].metricKeys;
                        const groupKey = `group-${groupIdx}`;

                        // Per-group aggregation settings
                        const aggSettings = streamAggregation[groupKey] || { mode: 'final', optimizeMetric: metricKeys[0] || 'loss' };
                        const { mode: aggregationMode, optimizeMetric } = aggSettings;

                        // Filter runs that have ANY series from this group
                        const groupRunsWithSeries = runsWithSeries.filter(run => {
                          return seriesGroup.some(({ seriesName }) => {
                            return metricKeys.some(key => {
                              const value = getMetricValue(run, key, { mode: aggregationMode, optimizeMetric, streamId: seriesName });
                              return value !== null && value !== undefined;
                            });
                          });
                        });

                        if (groupRunsWithSeries.length === 0) return null;

                        const groupTitle = seriesGroup.length === 1
                          ? seriesGroup[0].seriesName
                          : seriesGroup.map(s => s.seriesName).join(', ');

                        // Default to collapsed (only expanded if in expandedTables Set)
                        const isCollapsed = !expandedTables.has(groupKey);
                        const filteredGroupRuns = tableFilteredRuns[groupKey] || groupRunsWithSeries;

                        return (
                          <div key={groupKey} style={{ marginBottom: '24px', position: 'relative' }}>
                            <h4 style={{ margin: 0, marginBottom: '12px', fontSize: '14px', fontWeight: '600', paddingRight: '80px' }}>
                              {groupTitle}
                            </h4>
                            <CollapsibleButton
                              isCollapsed={isCollapsed}
                              onClick={() => toggleTableExpand(groupKey)}
                            />

                            {!isCollapsed && (
                              <>
                                {/* Per-group aggregation controls */}
                                <div style={{ marginBottom: '12px', display: 'inline-block' }}>
                                  <div style={{ display: 'flex', gap: '8px', alignItems: 'center', fontSize: '12px' }}>
                                    <label style={{ fontWeight: '500', minWidth: '75px' }}>Aggregation:</label>
                                    <select
                                      value={aggregationMode}
                                      onChange={(e) => setStreamAggregation({
                                        ...streamAggregation,
                                        [groupKey]: { ...aggSettings, mode: e.target.value }
                                      })}
                                      style={{ fontSize: '12px', padding: '2px 4px' }}
                                    >
                                      <option value="min">Min</option>
                                      <option value="max">Max</option>
                                      <option value="final">Final</option>
                                    </select>
                                  </div>
                                </div>
                                <RunFilter
                                  runs={groupRunsWithSeries}
                                  onFilterChange={(filtered) => setTableFilteredRuns({ ...tableFilteredRuns, [groupKey]: filtered })}
                                  aggregationMode={aggregationMode}
                                  optimizeMetric={optimizeMetric}
                                />
                                <table>
                              <thead>
                                <tr>
                                  <th style={{width: '30px'}}></th>
                                  <th>
                                    <input
                                      type="checkbox"
                                      checked={selectedRunIds.length === filteredGroupRuns.length && filteredGroupRuns.length > 0}
                                      onChange={(e) => {
                                        if (e.target.checked) {
                                          onRunSelectionChange(filteredGroupRuns.map(r => r.run_id));
                                        } else {
                                          onRunSelectionChange([]);
                                        }
                                      }}
                                    />
                                  </th>
                                  <th>Run ID</th>
                                  {metricKeys.map(metricKey => (
                                    <th key={metricKey}>{getDisplayName(metricKey)}</th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {filteredGroupRuns.map(run => {
                                  // Find which series from this group the run has
                                  const runSeriesName = seriesGroup.find(({ seriesName }) => {
                                    return run.structured_data?.[seriesName];
                                  })?.seriesName;

                                  return (
                                    <React.Fragment key={run.run_id}>
                                      <tr
                                        className={selectedRunIds.includes(run.run_id) ? 'selected' : ''}
                                        onClick={() => toggleRunSelection(run.run_id)}
                                      >
                                        <td onClick={(e) => e.stopPropagation()}>
                                          <button
                                            className="expand-btn"
                                            onClick={(e) => {
                                              e.stopPropagation();
                                              toggleExpandRun(run.run_id);
                                            }}
                                            disabled={!run.metadata?.config}
                                          >
                                            {run.metadata?.config ? (expandedRunIds.has(run.run_id) ? '▼' : '▶') : '-'}
                                          </button>
                                        </td>
                                        <td onClick={(e) => e.stopPropagation()}>
                                          <input
                                            type="checkbox"
                                            checked={selectedRunIds.includes(run.run_id)}
                                            onChange={() => toggleRunSelection(run.run_id)}
                                          />
                                        </td>
                                        <td title={run.run_id}>{run.name || run.run_id.substring(0, 20) + '...'}</td>
                                        {metricKeys.map(metricKey => (
                                          <td key={metricKey}>
                                            {formatMetricValue(metricKey, getMetricValue(run, metricKey, { mode: aggregationMode, optimizeMetric, streamId: runSeriesName }))}
                                          </td>
                                        ))}
                                      </tr>
                                      {expandedRunIds.has(run.run_id) && run.metadata?.config && (
                                        <tr className="config-row" key={`${run.run_id}-config`}>
                                          <td colSpan={3 + metricKeys.length}>
                                            <ConfigDisplay config={run.metadata?.config} />
                                          </td>
                                        </tr>
                                      )}
                                    </React.Fragment>
                                  );
                                })}
                              </tbody>
                            </table>
                              </>
                            )}
                          </div>
                        );
                      })}
                    </>
                  );
                })()}

                {/* Table Primitives - arbitrary tabular data logged by user */}
                {(() => {
                  // Find all table primitives across selected runs
                  const tablesData = {};

                  selectedRuns.forEach(run => {
                    if (!run.structured_data) return;

                    Object.entries(run.structured_data).forEach(([name, entries]) => {
                      const latestEntry = entries[entries.length - 1];
                      if (latestEntry.primitive_type === 'table') {
                        if (!tablesData[name]) {
                          tablesData[name] = [];
                        }
                        tablesData[name].push({
                          runId: run.run_id,
                          runName: run.name || run.run_id,
                          data: latestEntry.data
                        });
                      }
                    });
                  });

                  if (Object.keys(tablesData).length === 0) return null;

                  return (
                    <>
                      {Object.entries(tablesData).map(([tableName, runTables]) => {
                        const tableKey = `table-${tableName}`;
                        const isCollapsed = !expandedTables.has(tableKey);

                        return (
                          <div key={tableKey} style={{ marginBottom: '24px', marginTop: '12px', position: 'relative' }}>
                            <h4 style={{ margin: 0, marginBottom: '12px', fontSize: '14px', fontWeight: '600', paddingRight: '80px' }}>
                              {getDisplayName(tableName)}
                            </h4>
                            <CollapsibleButton
                              isCollapsed={isCollapsed}
                              onClick={() => toggleTableExpand(tableKey)}
                            />

                            {!isCollapsed && runTables.map((runTable) => {
                              const { runName, data } = runTable;
                              if (!data || !data.columns || !data.rows) return null;

                              return (
                                <div key={runTable.runId} style={{ marginBottom: '16px' }}>
                                  <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px', fontWeight: '500' }}>
                                    {runName}
                                  </div>
                                  <table>
                                    <thead>
                                      <tr>
                                        {data.columns.map((col, idx) => (
                                          <th key={idx}>{typeof col === 'string' ? getDisplayName(col) : getDisplayName(col.name || col)}</th>
                                        ))}
                                      </tr>
                                    </thead>
                                    <tbody>
                                      {data.rows.map((row, rowIdx) => (
                                        <tr key={rowIdx}>
                                          {row.map((cell, cellIdx) => (
                                            <td key={cellIdx}>
                                              {formatMetricValue(data.columns[cellIdx], cell)}
                                            </td>
                                          ))}
                                        </tr>
                                      ))}
                                    </tbody>
                                  </table>
                                </div>
                              );
                            })}
                          </div>
                        );
                      })}
                    </>
                  );
                })()}
              </>
            );
          })()}
        </div>
    </div>
  );
};
