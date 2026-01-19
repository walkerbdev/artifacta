import React, { useRef, useMemo, useState, useCallback } from 'react';
import { extractValues, getShortParamName } from '@/app/utils/comparisonPlotDiscovery';
import { getChartFont } from '@/app/hooks/useCanvasSetup';
import { useResponsiveCanvas } from '@/app/hooks/useResponsiveCanvas';
import { CHART_PADDING } from '@/core/utils/constants';

/**
 * Parallel Coordinates Chart
 * Visualizes relationships between hyperparameters and a selected metric across multiple runs
 * Features: smooth curves, color gradient by metric value, hover tooltips, metric aggregation
 */
const ParallelCoordinatesChart = ({ hyperparameters, availableMetrics, defaultMetric, data, runs }) => {
  const canvasRef = useRef(null);
  const [selectedMetric, setSelectedMetric] = useState(defaultMetric || availableMetrics?.[0]);
  const [aggregation, setAggregation] = useState('last');
  const [hoveredRun, setHoveredRun] = useState(null);

  // Re-aggregate data when aggregation method changes
  const aggregatedData = useMemo(() => {
    if (!runs) return data; // Fallback to pre-aggregated data if no raw runs

    return runs.map(run => ({
      run_id: run.run_id,
      name: run.name || run.run_id,
      hyperparams: extractValues(run, hyperparameters, 'last'), // Hyperparams don't aggregate
      metrics: extractValues(run, availableMetrics, aggregation) // Metrics use selected aggregation
    }));
  }, [runs, hyperparameters, availableMetrics, aggregation, data]);

  // Build axes: all hyperparameters + selected metric
  const axes = useMemo(() => {
    return [...hyperparameters, selectedMetric];
  }, [hyperparameters, selectedMetric]);

  // Color gradient function (purple/blue -> teal -> green, inspired by W&B)
  const getGradientColor = (normalizedValue) => {
    // Beautiful gradient: purple-blue (low) -> cyan -> green-yellow (high)
    if (normalizedValue < 0.5) {
      // 0.0 -> 0.5: purple-blue (#8B5CF6) to cyan (#06B6D4)
      const t = normalizedValue * 2; // normalize to 0-1
      const r = Math.round(139 + (6 - 139) * t);
      const g = Math.round(92 + (182 - 92) * t);
      const b = Math.round(246 + (212 - 246) * t);
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      // 0.5 -> 1.0: cyan (#06B6D4) to green (#10B981)
      const t = (normalizedValue - 0.5) * 2; // normalize to 0-1
      const r = Math.round(6 + (16 - 6) * t);
      const g = Math.round(182 + (185 - 182) * t);
      const b = Math.round(212 + (129 - 212) * t);
      return `rgb(${r}, ${g}, ${b})`;
    }
  };

  // Normalize data for each axis to 0-1 range
  const normalizedData = useMemo(() => {
    if (!aggregatedData || aggregatedData.length === 0 || !axes || axes.length === 0) return null;

    const ranges = {};

    // Calculate min/max for each axis
    axes.forEach(axis => {
      const isHyperparam = hyperparameters.includes(axis);
      const values = aggregatedData
        .map(d => isHyperparam ? d.hyperparams[axis] : d.metrics[axis])
        .filter(v => v !== null && v !== undefined);

      if (values.length === 0) {
        ranges[axis] = { min: 0, max: 1, type: 'numeric' };
        return;
      }

      // Handle numeric values
      if (values.every(v => typeof v === 'number')) {
        ranges[axis] = {
          min: Math.min(...values),
          max: Math.max(...values),
          type: 'numeric'
        };
      } else {
        // Handle categorical - assign indices
        const uniqueValues = [...new Set(values)];
        ranges[axis] = {
          min: 0,
          max: uniqueValues.length - 1,
          type: 'categorical',
          categories: uniqueValues
        };
      }
    });

    // Get metric values for color gradient
    const metricValues = aggregatedData.map(run => run.metrics[selectedMetric]).filter(v => v !== null && v !== undefined);
    const metricMin = Math.min(...metricValues);
    const metricMax = Math.max(...metricValues);
    const metricSpan = metricMax - metricMin;

    // Normalize each run's data
    return aggregatedData.map((run) => {
      const normalizedValues = axes.map(axis => {
        const isHyperparam = hyperparameters.includes(axis);
        const value = isHyperparam ? run.hyperparams[axis] : run.metrics[axis];

        if (value === null || value === undefined) return null;

        const range = ranges[axis];
        if (range.type === 'categorical') {
          const index = range.categories.indexOf(value);
          return index / (range.max || 1);
        }

        // Numeric normalization
        const span = range.max - range.min;
        if (span === 0) return 0.5; // All values same
        return (value - range.min) / span;
      });

      // Color by selected metric value
      const metricValue = run.metrics[selectedMetric];
      let color;
      if (metricValue === null || metricValue === undefined || metricSpan === 0) {
        color = '#999';
      } else {
        const normalizedMetric = (metricValue - metricMin) / metricSpan;
        color = getGradientColor(normalizedMetric);
      }

      return {
        ...run,
        normalizedValues,
        ranges,
        color,
        metricValue
      };
    });
  }, [aggregatedData, axes, hyperparameters, selectedMetric]);

  const drawChart = useCallback((width, height) => {
    const canvas = canvasRef.current;
    if (!canvas || !normalizedData || !axes || axes.length === 0) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);

    const padding = { ...CHART_PADDING, left: 100, right: 100, top: 80 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    // Clear
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    // Calculate axis positions
    const numAxes = axes.length;
    const axisSpacing = plotWidth / (numAxes - 1);

    // Draw axes
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    axes.forEach((axis, i) => {
      const x = padding.left + i * axisSpacing;
      const isLastAxis = i === axes.length - 1;

      if (isLastAxis) {
        // Last axis: draw as color gradient bar instead of line
        const barWidth = 20;
        const barX = x - barWidth / 2;

        // Draw gradient bar
        const gradient = ctx.createLinearGradient(0, padding.top + plotHeight, 0, padding.top);
        gradient.addColorStop(0, getGradientColor(0));
        gradient.addColorStop(0.5, getGradientColor(0.5));
        gradient.addColorStop(1, getGradientColor(1));

        ctx.fillStyle = gradient;
        ctx.fillRect(barX, padding.top, barWidth, plotHeight);

        // Border
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.strokeRect(barX, padding.top, barWidth, plotHeight);
      } else {
        // Regular axis line
        ctx.beginPath();
        ctx.moveTo(x, padding.top);
        ctx.lineTo(x, padding.top + plotHeight);
        ctx.stroke();
      }

      // Min/Max labels first (so they render below title)
      ctx.font = getChartFont('tick');
      ctx.fillStyle = '#666';

      if (normalizedData.length > 0) {
        const range = normalizedData[0].ranges[axis];
        if (range && range.type === 'numeric') {
          if (isLastAxis) {
            // For color bar: labels on right side, aligned to top/bottom of bar
            ctx.textAlign = 'left';
            ctx.textBaseline = 'bottom';
            ctx.fillText(range.max.toFixed(2), x + 15, padding.top);
            ctx.textBaseline = 'top';
            ctx.fillText(range.min.toFixed(2), x + 15, padding.top + plotHeight);
          } else {
            // Regular axes: labels centered on axis
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.fillText(range.min.toFixed(2), x, padding.top + plotHeight + 12);
            ctx.textBaseline = 'bottom';
            ctx.fillText(range.max.toFixed(2), x, padding.top - 5);
          }
        }
      }

      // Axis label (horizontal, centered above max value)
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillStyle = '#333';
      ctx.font = getChartFont('axisLabel');

      // Use short name (last segment after period) and capitalize
      const shortName = getShortParamName(axis);
      const label = shortName.charAt(0).toUpperCase() + shortName.slice(1);

      ctx.fillText(label, x, padding.top - 20);
    });

    // Draw smooth curves for each run
    normalizedData.forEach((run) => {
      const isHovered = hoveredRun === run.run_id;

      ctx.strokeStyle = run.color;
      ctx.lineWidth = isHovered ? 3 : 1.5;
      ctx.globalAlpha = isHovered ? 1.0 : 0.4;

      ctx.beginPath();

      run.normalizedValues.forEach((normalizedValue, axisIdx) => {
        if (normalizedValue === null) return;

        const x = padding.left + axisIdx * axisSpacing;
        const y = padding.top + plotHeight - (normalizedValue * plotHeight);

        if (axisIdx === 0) {
          ctx.moveTo(x, y);
        } else {
          // Draw smooth curve to this point (Bezier curve)
          const prevX = padding.left + (axisIdx - 1) * axisSpacing;
          const prevY = padding.top + plotHeight - (run.normalizedValues[axisIdx - 1] * plotHeight);

          const controlPointOffset = axisSpacing / 2;
          const cp1x = prevX + controlPointOffset;
          const cp1y = prevY;
          const cp2x = x - controlPointOffset;
          const cp2y = y;

          ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, x, y);
        }
      });

      ctx.stroke();
    });

    ctx.globalAlpha = 1.0;
  }, [normalizedData, axes, hoveredRun]);

  useResponsiveCanvas(canvasRef, drawChart);

  // Throttled mouse move handler for performance
  const throttledMouseMoveRef = useRef(null);

  const handleMouseMove = useCallback((e) => {
    // Throttle to ~60fps
    if (throttledMouseMoveRef.current) return;

    throttledMouseMoveRef.current = setTimeout(() => {
      throttledMouseMoveRef.current = null;
    }, 16);

    const canvas = canvasRef.current;
    if (!canvas || !normalizedData) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const displayWidth = canvas.clientWidth || 800;
    const displayHeight = canvas.clientHeight || 400;
    const padding = { ...CHART_PADDING, left: 100, right: 100, top: 80 };
    const plotWidth = displayWidth - padding.left - padding.right;
    const plotHeight = displayHeight - padding.top - padding.bottom;
    const numAxes = axes.length;
    const axisSpacing = plotWidth / (numAxes - 1);

    // Find closest line to mouse
    let closestRun = null;
    let closestDistance = Infinity;

    for (const run of normalizedData) {
      // Check distance to each Bezier curve segment
      for (let i = 0; i < run.normalizedValues.length - 1; i++) {
        if (run.normalizedValues[i] === null || run.normalizedValues[i + 1] === null) continue;

        const x1 = padding.left + i * axisSpacing;
        const y1 = padding.top + plotHeight - (run.normalizedValues[i] * plotHeight);
        const x2 = padding.left + (i + 1) * axisSpacing;
        const y2 = padding.top + plotHeight - (run.normalizedValues[i + 1] * plotHeight);

        // Bezier control points (same as drawing code)
        const controlPointOffset = axisSpacing / 2;
        const cp1x = x1 + controlPointOffset;
        const cp1y = y1;
        const cp2x = x2 - controlPointOffset;
        const cp2y = y2;

        // Sample 10 points along the Bezier curve for hover detection
        for (let t = 0; t <= 1; t += 0.1) {
          // Cubic Bezier formula optimized: B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
          const oneMinusT = 1 - t;
          const oneMinusT2 = oneMinusT * oneMinusT;
          const oneMinusT3 = oneMinusT2 * oneMinusT;
          const t2 = t * t;
          const t3 = t2 * t;

          const curveX = oneMinusT3 * x1 + 3 * oneMinusT2 * t * cp1x + 3 * oneMinusT * t2 * cp2x + t3 * x2;
          const curveY = oneMinusT3 * y1 + 3 * oneMinusT2 * t * cp1y + 3 * oneMinusT * t2 * cp2y + t3 * y2;

          // Skip sqrt for comparison (compare squared distances)
          const dx = mouseX - curveX;
          const dy = mouseY - curveY;
          const distanceSquared = dx * dx + dy * dy;

          if (distanceSquared < closestDistance * closestDistance) {
            closestDistance = Math.sqrt(distanceSquared);
            closestRun = run;

            // Early exit if we found a very close match
            if (distanceSquared < 4) break; // Within 2px
          }
        }

        // Early exit if we found a very close match
        if (closestDistance < 2) break;
      }
    }

    // Only hover if within 10px
    if (closestDistance < 10 && closestRun) {
      setHoveredRun(closestRun.run_id);
      canvas.style.cursor = 'pointer';
      canvas.title = `Run: ${closestRun.run_id}\n${selectedMetric}: ${closestRun.metricValue?.toFixed(4) || 'N/A'}`;
    } else {
      setHoveredRun(null);
      canvas.style.cursor = 'default';
      canvas.title = '';
    }
  }, [normalizedData, axes, selectedMetric]);


  if (!aggregatedData || aggregatedData.length === 0 || !hyperparameters || hyperparameters.length === 0) {
    return (
      <div style={{ padding: '20px', textAlign: 'center', color: '#666' }}>
        No data available for parallel coordinates plot.
        Select at least 2 runs with varying hyperparameters.
      </div>
    );
  }

  return (
    <div style={{ padding: '10px' }}>
      <div style={{ width: '100%', display: 'flex', flexDirection: 'column' }}>
        {/* Controls */}
        <div style={{ padding: '0 16px 16px 16px', display: 'flex', gap: '20px', alignItems: 'center', justifyContent: 'center' }}>
        <div>
          <label style={{ fontSize: '13px', color: '#666', marginRight: '8px' }}>
            Metric:
          </label>
          <select
            value={selectedMetric || ''}
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
          className="viz-canvas-parallel-coordinates"
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHoveredRun(null)}
        />
      </div>
    </div>
  );
};

export default ParallelCoordinatesChart;
