import React, { useRef, useState, useCallback, useMemo, useEffect } from 'react';
import { calculateParameterCorrelation, getShortParamName } from '@/app/utils/comparisonPlotDiscovery';
import { drawTopLegend, getChartFont } from '@/app/hooks/useCanvasSetup';
import { useResponsiveCanvas } from '@/app/hooks/useResponsiveCanvas';
import { CHART_PADDING } from '@/core/utils/constants';

/**
 * Parameter Correlation Chart
 * Shows correlation between hyperparameters and metrics using Pearson correlation
 *
 * Displays:
 * - Correlation: Linear relationship between parameter and metric (-1 to 1)
 * - Visual bars: Green for positive, red for negative correlation
 * @param {object} props - Component props
 * @param {string[]} props.hyperparameters - Array of hyperparameter names
 * @param {string[]} props.availableMetrics - Array of available metric names
 * @param {string} props.defaultMetric - Default metric to display
 * @param {object} props.importance - Pre-calculated parameter importance data
 * @param {Array} props.runs - Array of run objects with hyperparameter and metric data
 * @returns {React.ReactElement} The rendered component
 */
const ParameterCorrelationChart = ({ hyperparameters, availableMetrics, defaultMetric, importance: defaultImportance, runs }) => {
  const canvasRef = useRef(null);
  const [selectedMetric, setSelectedMetric] = useState(() => {
    return defaultMetric || availableMetrics?.[0] || '';
  });
  const [aggregation, setAggregation] = useState('last');

  // Update selectedMetric if it becomes null/invalid
  useEffect(() => {
    if (!selectedMetric && availableMetrics?.length > 0) {
      setSelectedMetric(availableMetrics[0]);
    }
  }, [selectedMetric, availableMetrics]);

  // Recalculate correlation when aggregation changes
  const importance = useMemo(() => {
    if (!runs || aggregation === 'last') return defaultImportance; // Use pre-calculated if default

    // Recalculate with new aggregation
    return calculateParameterCorrelation(runs, hyperparameters, availableMetrics, aggregation);
  }, [runs, hyperparameters, availableMetrics, aggregation, defaultImportance]);

  /**
   * Draws the parameter correlation chart on canvas
   * @param {number} width - Canvas width
   * @param {number} height - Canvas height
   * @returns {void}
   */
  const drawChart = useCallback((width, height) => {
    const canvas = canvasRef.current;
    if (!canvas || !importance || !selectedMetric || !hyperparameters) return;

    // Setup canvas with DPR
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);

    // Clear
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    // Draw legend at top
    const legendItems = [
      { label: 'Positive correlation', color: 'rgba(16, 185, 129, 0.7)' },
      { label: 'Negative correlation', color: 'rgba(239, 68, 68, 0.7)' }
    ];
    const legendHeight = drawTopLegend(ctx, legendItems, width, 10, {
      leftMargin: 150,
      rightMargin: 80
    });

    const padding = { ...CHART_PADDING, left: 150, right: 80, top: 10 + legendHeight + 10 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    // Get importance data for selected metric
    const metricData = importance[selectedMetric];
    if (!metricData) return;

    // Sort parameters by absolute correlation (descending)
    const sortedParams = hyperparameters
      .map(param => ({
        param,
        importance: metricData[param]?.importance || 0,
        correlation: metricData[param]?.correlation || 0
      }))
      .sort((a, b) => b.importance - a.importance);

    const barHeight = plotHeight / sortedParams.length;
    const maxBarHeight = Math.min(barHeight * 0.8, 40);

    // Draw horizontal bars
    sortedParams.forEach((item, i) => {
      const y = padding.top + i * barHeight + (barHeight - maxBarHeight) / 2;
      const barLength = Math.abs(item.correlation) * plotWidth;

      // Color based on correlation (positive = green, negative = red)
      const absCorr = Math.abs(item.correlation);
      if (item.correlation > 0) {
        ctx.fillStyle = `rgba(16, 185, 129, ${0.3 + absCorr * 0.7})`; // Green
      } else {
        ctx.fillStyle = `rgba(239, 68, 68, ${0.3 + absCorr * 0.7})`; // Red
      }

      // Draw bar
      ctx.fillRect(padding.left, y, barLength, maxBarHeight);

      // Draw border
      ctx.strokeStyle = '#333';
      ctx.lineWidth = 1;
      ctx.strokeRect(padding.left, y, barLength, maxBarHeight);

      // Parameter label (left side)
      ctx.fillStyle = '#333';
      ctx.font = getChartFont('axisLabel');
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      const shortName = getShortParamName(item.param);
      const paramLabel = shortName.charAt(0).toUpperCase() + shortName.slice(1);
      ctx.fillText(paramLabel, padding.left - 10, y + maxBarHeight / 2);

      // Correlation value (right side of bar)
      ctx.fillStyle = '#666';
      ctx.font = getChartFont('tick');
      ctx.textAlign = 'left';
      const corrSign = item.correlation > 0 ? '+' : '';
      ctx.fillText(`${corrSign}${item.correlation.toFixed(2)}`, padding.left + barLength + 5, y + maxBarHeight / 2);
    });
  }, [importance, selectedMetric, hyperparameters]);

  useResponsiveCanvas(canvasRef, drawChart);


  if (!importance || !availableMetrics || availableMetrics.length === 0) {
    return (
      <div style={{ padding: '20px', textAlign: 'center', color: '#666' }}>
        No data available for parameter importance.
        Need at least 3 runs with varying hyperparameters.
      </div>
    );
  }

  return (
    <div style={{ padding: '10px' }}>
      <div style={{ width: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Title */}
      <div style={{ padding: '12px 16px', textAlign: 'center' }}>
        <h4 style={{ margin: 0, fontSize: '14px', fontWeight: '600', color: '#333' }}>
          Parameter Correlation
        </h4>
      </div>

      {/* Controls */}
      <div style={{ padding: '0 16px 16px 16px', display: 'flex', gap: '20px', alignItems: 'center', justifyContent: 'center' }}>
        <div>
          <label style={{ fontSize: '13px', color: '#666', marginRight: '8px' }}>
            Metric:
          </label>
          <select
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value)}
            style={{
              padding: '4px 8px',
              fontSize: '13px',
              border: '1px solid #d0d0d0',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            {availableMetrics.map(metric => (
              <option key={metric} value={metric}>
                {metric}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label style={{ fontSize: '13px', color: '#666', marginRight: '8px' }}>
            Aggregation:
          </label>
          <select
            value={aggregation}
            onChange={(e) => setAggregation(e.target.value)}
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

      <canvas
        ref={canvasRef}
        className="viz-canvas-parameter-importance"
        style={{ width: '100%', height: '100%', display: 'block' }}
      />
      </div>
    </div>
  );
};

export default ParameterCorrelationChart;
