import React, { useMemo, useState } from 'react';
import { detectSweep } from '@/app/utils/sweepDetection';
import ParallelCoordinatesChart from '@/app/components/visualizations/plots/ParallelCoordinatesChart';
import ParameterCorrelationChart from '@/app/components/visualizations/plots/ParameterCorrelationChart';
import ScatterPlot from '@/app/components/visualizations/plots/ScatterPlot';
import { DraggableVisualization } from '@/app/components/visualizations/DraggableVisualization';
import { useLayoutManager } from '@/app/hooks';
import { calculateParameterImportance } from '@/app/utils/comparisonPlotDiscovery';
import './SweepsTab.scss';

/**
 * Sweeps Tab
 *
 * Analyzes selected runs to detect valid hyperparameter sweeps and visualizes:
 * - Parallel Coordinates: Multi-dimensional view of params → metrics
 * - Scatter Plots: Swept parameter vs target metric
 *
 * Only renders if selected runs form a valid sweep (same keys, one varying param)
 */
export const SweepsTab = ({ runs, selectedRunIds }) => {
  const [selectedMetric, setSelectedMetric] = useState(null);
  const [scatterMetrics, setScatterMetrics] = useState({});
  const [scatterAggregations, setScatterAggregations] = useState({});
  const [hiddenVisualizations, setHiddenVisualizations] = useState(new Set());

  // Layout manager for draggable/resizable charts
  const {
    dragTransforms,
    draggingKey,
    handleDragStart,
    handleDrag,
    handleDragEnd,
    handleResize,
    registerElement,
    registerContainer,
    customLabels,
    updateLabels
  } = useLayoutManager();

  const toggleVisualizationVisibility = (vizKey) => {
    setHiddenVisualizations(prev => {
      const next = new Set(prev);
      if (next.has(vizKey)) {
        next.delete(vizKey);
      } else {
        next.add(vizKey);
      }
      return next;
    });
  };

  // Detect sweep from selected runs
  const sweepData = useMemo(() => {
    if (!runs || !selectedRunIds || selectedRunIds.length === 0) {
      return null;
    }

    const selectedRuns = runs.filter(r => selectedRunIds.includes(r.run_id));
    const result = detectSweep(selectedRuns);

    return result;
  }, [runs, selectedRunIds]);

  // Calculate parameter correlations for all varying parameters
  const parameterCorrelations = useMemo(() => {
    if (!sweepData?.valid || !runs || selectedRunIds.length < 3) {
      return null;
    }
    // Get original runs (not the transformed sweep runs) - calculateParameterImportance needs full run structure
    const originalRuns = runs.filter(r => selectedRunIds.includes(r.run_id));

    // Calculate correlation for all varying parameters
    const varyingParamNames = sweepData.varyingParams.map(p => p.name);
    return calculateParameterImportance(
      originalRuns,
      varyingParamNames,
      sweepData.availableMetrics,
      'last'
    );
  }, [sweepData, runs, selectedRunIds]);

  // Auto-select first metric
  React.useEffect(() => {
    if (sweepData?.valid && sweepData.availableMetrics.length > 0 && !selectedMetric) {
      setSelectedMetric(sweepData.availableMetrics[0]);
    }
  }, [sweepData, selectedMetric]);

  // No runs selected
  if (!selectedRunIds || selectedRunIds.length === 0) {
    return null;
  }

  // Invalid sweep - return empty
  if (!sweepData || !sweepData.valid) {
    return null;
  }

  // Valid sweep - render visualizations
  const { runs: sweepRuns, availableMetrics } = sweepData;

  return (
    <div className="sweeps-tab">
      {/* Visualizations - Draggable & Resizable */}
      <div className="sweeps-visualizations" ref={registerContainer}>
        {/* Scatter Plots: One for each varying parameter (numeric only) */}
        {sweepData.varyingParams.filter(p => p.isNumeric).map((param) => {
          const scatterMetric = scatterMetrics[param.name] || selectedMetric;
          const scatterAggregation = scatterAggregations[param.name] || 'last';
          const vizKey = `sweep-scatter-${param.name}-${scatterMetric}`;
          if (hiddenVisualizations.has(vizKey)) return null;

          return (
            <DraggableVisualization
              key={vizKey}
              visualizationKey={vizKey}
              title={`Impact of ${param.name} on ${scatterMetric}`}
              onClose={toggleVisualizationVisibility}
              onDragStart={handleDragStart}
              onDrag={handleDrag}
              onDragEnd={handleDragEnd}
              onResize={handleResize}
              dragTransform={dragTransforms[vizKey]}
              isDragging={draggingKey === vizKey}
              registerElement={registerElement}
              customLabels={customLabels[vizKey]}
              onUpdateLabels={(labels) => updateLabels(vizKey, labels)}
              chartType="scatter"
            >
              <div style={{ padding: '0 16px 16px 16px', display: 'flex', gap: '20px', alignItems: 'center', justifyContent: 'center' }}>
                <div>
                  <label style={{ fontSize: '13px', color: '#666', marginRight: '8px' }}>
                    Metric:
                  </label>
                  <select
                    value={scatterMetric || ''}
                    onChange={(e) => {
                      setScatterMetrics(prev => ({ ...prev, [param.name]: e.target.value }));
                    }}
                    style={{
                      padding: '4px 8px',
                      fontSize: '13px',
                      border: '1px solid #d0d0d0',
                      borderRadius: '4px',
                      cursor: 'pointer'
                    }}
                  >
                    {availableMetrics.map(metric => (
                      <option key={metric} value={metric}>{metric}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label style={{ fontSize: '13px', color: '#666', marginRight: '8px' }}>
                    Aggregation:
                  </label>
                  <select
                    value={scatterAggregation}
                    onChange={(e) => {
                      setScatterAggregations(prev => ({ ...prev, [param.name]: e.target.value }));
                    }}
                    style={{
                      padding: '4px 8px',
                      fontSize: '13px',
                      border: '1px solid #d0d0d0',
                      borderRadius: '4px',
                      cursor: 'pointer'
                    }}
                  >
                    <option value="last">Last</option>
                    <option value="max">Max</option>
                    <option value="min">Min</option>
                  </select>
                </div>
              </div>

            <ScatterPlot
              data={{
                x: sweepRuns.map(r => r.config?.[param.name] ?? null),
                y: sweepRuns.map(r => r.metrics?.[scatterMetric] ?? null),
                xLabel: param.name,
                yLabel: scatterMetric
              }}
            />
            </DraggableVisualization>
          );
        })}

        {/* Parallel Coordinates */}
        {!hiddenVisualizations.has('sweep-parallel-coordinates') && (
          <DraggableVisualization
            visualizationKey="sweep-parallel-coordinates"
            title="Parallel Coordinates"
            onClose={toggleVisualizationVisibility}
            onDragStart={handleDragStart}
            onDrag={handleDrag}
            onDragEnd={handleDragEnd}
            onResize={handleResize}
            dragTransform={dragTransforms['sweep-parallel-coordinates']}
            isDragging={draggingKey === 'sweep-parallel-coordinates'}
            registerElement={registerElement}
            customLabels={customLabels['sweep-parallel-coordinates']}
            onUpdateLabels={(labels) => updateLabels('sweep-parallel-coordinates', labels)}
            chartType="parallel_coordinates"
          >
          <ParallelCoordinatesChart
            hyperparameters={sweepData.varyingParams.map(p => p.name)}
            availableMetrics={availableMetrics}
            defaultMetric={selectedMetric}
            data={sweepRuns.map(r => ({
              run_id: r.run_id,
              name: r.name,
              hyperparams: Object.fromEntries(
                sweepData.varyingParams.map(p => [p.name, r.config[p.name]])
              ),
              metrics: r.metrics
            }))}
            runs={null}
          />
          </DraggableVisualization>
        )}

        {/* Parameter Correlation Chart */}
        {!hiddenVisualizations.has('sweep-parameter-correlation') && parameterCorrelations && (
          <DraggableVisualization
            visualizationKey="sweep-parameter-correlation"
            title="Parameter Correlation"
            onClose={toggleVisualizationVisibility}
            onDragStart={handleDragStart}
            onDrag={handleDrag}
            onDragEnd={handleDragEnd}
            onResize={handleResize}
            dragTransform={dragTransforms['sweep-parameter-correlation']}
            isDragging={draggingKey === 'sweep-parameter-correlation'}
            registerElement={registerElement}
            customLabels={customLabels['sweep-parameter-correlation']}
            onUpdateLabels={(labels) => updateLabels('sweep-parameter-correlation', labels)}
            chartType="parameter_importance"
          >
            <ParameterCorrelationChart
              hyperparameters={sweepData.varyingParams.map(p => p.name)}
              availableMetrics={availableMetrics}
              defaultMetric={selectedMetric}
              importance={parameterCorrelations}
              runs={runs.filter(r => selectedRunIds.includes(r.run_id))}
            />
          </DraggableVisualization>
        )}
      </div>
    </div>
  );
};
