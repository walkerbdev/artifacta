import React, { useRef, useCallback } from 'react';
import { useCanvasTooltip } from '../../../hooks/useCanvasTooltip';
import { useResponsiveCanvas } from '../../../hooks/useResponsiveCanvas';
import PlotTooltip from '../shared/PlotTooltip';
import { drawYAxisWithGrid, getChartFont } from '../../../hooks/useCanvasSetup';
import { getChartColor, CHART_PADDING } from '../../../../core/utils/constants';
import { formatYAxisValue } from '../../../../core/utils/formatters';

/**
 * Line Plot component for time series visualization
 *
 * Renders multi-series line charts with interactive tooltips, auto-scaled axes,
 * and responsive resizing. Optimized for training metrics (loss curves, accuracy over time).
 *
 * Features:
 * - Multi-series overlay (multiple lines on one chart with color coding)
 * - Interactive tooltips showing exact values on hover
 * - Auto-scaled Y-axis based on data range
 * - Grid lines for easier value reading
 * - Legend showing series names and colors
 * - HiDPI (Retina) display support
 * - Responsive to container size changes
 *
 * Data format:
 * ```
 * {
 *   xLabel: "Epoch",         // X-axis label
 *   datasets: [
 *     {
 *       label: "train_loss",  // Series name
 *       data: [
 *         { x: 0, y: 0.5 },   // Individual points
 *         { x: 1, y: 0.3 },
 *         { x: 2, y: 0.2 }
 *       ]
 *     },
 *     { label: "val_loss", data: [...] }
 *   ]
 * }
 * ```
 *
 * @param {object} props - Component props
 * @param {object} props.data - Plot data with datasets and labels
 * @param {string} props.data.xLabel - X-axis label (e.g., "Epoch", "Step")
 * @param {Array<object>} props.data.datasets - Array of series to plot
 * @param {string} [props.title] - Optional plot title
 * @returns {React.ReactElement} Canvas-based line plot with interactive tooltip
 */
const LinePlot = ({ data, title }) => {
  const canvasRef = useRef(null);
  const plotDataRef = useRef(null);

  const drawLinePlot = useCallback((width, height) => {
    try {
      if (!data || !data.datasets) return;

      const { datasets, xLabel = 'X' } = data;

      const canvas = canvasRef.current;
      if (!canvas) return;

      // Setup canvas - only update if dimensions changed to prevent resize loops
      const dpr = window.devicePixelRatio || 1;
      const targetWidth = Math.floor(width * dpr);
      const targetHeight = Math.floor(height * dpr);


      if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
        canvas.width = targetWidth;
        canvas.height = targetHeight;
      }

      const ctx = canvas.getContext('2d');
      // CRITICAL: Reset transform before scaling to prevent cumulative scaling
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.scale(dpr, dpr);

      // Clear canvas
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, width, height);

      // Canvas is JUST the plot area - legend is HTML outside
      const padding = CHART_PADDING;
      const chartWidth = width - padding.left - padding.right;
      const chartHeight = height - padding.top - padding.bottom;

      // Collect all points for axis scaling
      const allPoints = datasets.flatMap(ds => ds.data || []);
      if (allPoints.length === 0) return;

      // Check if x-axis is categorical (strings) or numeric
      const firstX = allPoints[0].x;
      const isCategorical = typeof firstX === 'string' || isNaN(Number(firstX));

      let xMin, xMax, xLabels;
      let processedDatasets = datasets; // Use original datasets by default

      if (isCategorical) {
        // For categorical x-axis, extract unique labels and map to indices
        xLabels = [...new Set(allPoints.map(p => p.x))];
        xMin = 0;
        xMax = xLabels.length - 1;

        // Create NEW datasets with remapped x values (don't mutate original)
        processedDatasets = datasets.map(ds => ({
          ...ds,
          data: ds.data.map(point => ({
            ...point,
            xNumeric: xLabels.indexOf(point.x),
            xOriginal: point.x
          }))
        }));
      } else {
        // Numeric x-axis
        const xValues = allPoints.map(p => p.x).filter(v => v != null && !isNaN(v));
        xMin = Math.min(...xValues);
        xMax = Math.max(...xValues);
      }

      const yValues = allPoints.map(p => p.y).filter(v => v != null && !isNaN(v));
      if (yValues.length === 0) return;

      const yMin = Math.min(...yValues);
      const yMax = Math.max(...yValues);

      // Add 5% padding to y-axis, but handle edge case where yMin === yMax
      const yRange = yMax - yMin;
      const yPadding = yRange > 0 ? yRange * 0.05 : Math.abs(yMax) * 0.1 || 1;
      const yAxisMin = yMin - yPadding;
      const yAxisMax = yMax + yPadding;

      // Draw axes
      ctx.strokeStyle = '#333';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(padding.left, padding.top);
      ctx.lineTo(padding.left, height - padding.bottom);
      ctx.lineTo(width - padding.right, height - padding.bottom);
      ctx.stroke();

      // Draw Y-axis with grid lines and labels
      drawYAxisWithGrid(
        ctx,
        width,
        height,
        padding,
        yAxisMin,
        yAxisMax,
        (val) => formatYAxisValue(val, title),
        5
      );

      // Draw x-axis tick labels
      const baseY = height - padding.bottom + 15;
      let xAxisLabels, xPositions;

      if (isCategorical && xLabels) {
        // Categorical labels
        xAxisLabels = xLabels;
        xPositions = xLabels.map((_, i) =>
          xLabels.length > 1
            ? padding.left + (i / (xLabels.length - 1)) * chartWidth
            : padding.left + chartWidth / 2
        );
      } else {
        // Numeric tick labels - use fewer ticks to prevent overlap with rotated labels
        const numXTicks = Math.min(6, Math.max(2, Math.floor(chartWidth / 80)));
        xAxisLabels = [];
        xPositions = [];
        for (let i = 0; i < numXTicks; i++) {
          const ratio = numXTicks > 1 ? i / (numXTicks - 1) : 0.5;
          const xVal = xMin + ratio * (xMax - xMin);
          xAxisLabels.push(xVal.toFixed(1));
          xPositions.push(padding.left + ratio * chartWidth);
        }
      }

      // Draw x-axis labels horizontally
      ctx.fillStyle = '#666';
      ctx.font = getChartFont('tick');
      ctx.textAlign = 'center';
      xAxisLabels.forEach((label, i) => {
        ctx.fillText(label, xPositions[i], baseY);
      });

      // Y-axis label (centered on chart area, not full height)
      ctx.save();
      const chartCenterY = padding.top + chartHeight / 2;
      ctx.translate(20, chartCenterY);
      ctx.rotate(-Math.PI / 2);
      ctx.fillStyle = '#333';
      ctx.font = getChartFont('axisLabel');
      ctx.textAlign = 'center';
      ctx.fillText('Value', 0, 0);
      ctx.restore();

      // Draw each dataset
      processedDatasets.forEach((dataset, datasetIdx) => {
        const { data: points } = dataset;
        if (!points || points.length === 0) return;

        const color = getChartColor(datasetIdx);

        // Draw line
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();

        points.forEach((point, i) => {
          const xValue = isCategorical ? point.xNumeric : point.x;
          const x = padding.left + ((xValue - xMin) / (xMax - xMin)) * chartWidth;
          const y = padding.top + chartHeight - ((point.y - yAxisMin) / (yAxisMax - yAxisMin)) * chartHeight;

          if (i === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        });

        ctx.stroke();

        // Draw points
        ctx.fillStyle = color;
        points.forEach(point => {
          const xValue = isCategorical ? point.xNumeric : point.x;
          const x = padding.left + ((xValue - xMin) / (xMax - xMin)) * chartWidth;
          const y = padding.top + chartHeight - ((point.y - yAxisMin) / (yAxisMax - yAxisMin)) * chartHeight;

          ctx.beginPath();
          ctx.arc(x, y, 3, 0, Math.PI * 2);
          ctx.fill();
        });
      });

      // X-axis label (positioned at bottom)
      ctx.fillStyle = '#333';
      ctx.font = getChartFont('axisLabel');
      ctx.textAlign = 'center';
      ctx.fillText(xLabel, width / 2, height - 10);

      // Store data for tooltip
      plotDataRef.current = {
        plotArea: {
          left: padding.left,
          top: padding.top,
          width: chartWidth,
          height: chartHeight,
        },
        datasets: processedDatasets,
        xMin,
        xMax,
        yMin: yAxisMin,
        yMax: yAxisMax,
        isCategorical,
        xLabels,
        xLabel,
      };

      // Store plot data for tooltip
      // No return needed - container controls size, plot adapts
    } catch (error) {
      console.error('[LinePlot] Error rendering:', error, 'Data:', data);
    }
  }, [data, title]);

  // Use responsive canvas hook - handles sizing and redraw automatically
  useResponsiveCanvas(canvasRef, drawLinePlot);

  // Tooltip callback - finds nearest point on any dataset
  const getTooltipData = useCallback((canvasX, canvasY, searchRadius) => {
    const pd = plotDataRef.current;
    if (!pd || !pd.datasets) return null;

    const { plotArea, datasets, xMin, xMax, isCategorical, xLabel } = pd;

    // Check if within plot area horizontally
    const relX = (canvasX - plotArea.left) / plotArea.width;
    if (relX < 0 || relX > 1) return null;

    // Find nearest point across all datasets
    let nearestX = null;
    let minDist = searchRadius;

    datasets.forEach(ds => {
      if (!ds.data) return;
      ds.data.forEach(point => {
        const px = isCategorical ? point.xNumeric : point.x;
        if (px == null) return;

        const pointX = plotArea.left + ((px - xMin) / (xMax - xMin)) * plotArea.width;
        const dist = Math.abs(canvasX - pointX);

        if (dist < minDist) {
          minDist = dist;
          nearestX = isCategorical ? point.xOriginal : point.x;
        }
      });
    });

    if (nearestX === null) return null;

    // Gather all dataset values at this X coordinate
    const values = {};
    datasets.forEach(ds => {
      const match = ds.data?.find(p =>
        isCategorical ? p.xOriginal === nearestX : p.x === nearestX
      );
      if (match) {
        values[ds.label] = match.y;
      }
    });

    return {
      type: 'series',
      content: {
        index: nearestX,
        indexLabel: xLabel || 'x',
        values,
      },
    };
  }, []);

  const tooltip = useCanvasTooltip({
    canvasRef,
    getTooltipData,
    searchRadius: 30,
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', width: '100%', height: '100%' }}>
      {/* HTML Legend - auto-wraps and expands */}
      {data?.datasets && data.datasets.length > 0 && (
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '12px',
          padding: '10px 10px 5px 10px',
          fontSize: '13px',
          flexShrink: 0
        }}>
          {data.datasets.map((ds, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <div style={{
                width: '16px',
                height: '3px',
                backgroundColor: getChartColor(i),
                borderRadius: '1px'
              }} />
              <span>{ds.label}</span>
            </div>
          ))}
        </div>
      )}

      {/* Canvas - simple approach */}
      <div style={{ flex: 1, minHeight: 0, padding: '0 10px 10px 10px' }}>
        <canvas
          key={title} // Force new canvas element per unique title to prevent DOM reuse
          ref={canvasRef}
          className="viz-canvas-comparison-scatter"
          style={{
            width: '100%',
            height: '100%',
            maxWidth: '100%',
            maxHeight: '100%',
            display: 'block',
            cursor: 'crosshair'
          }}
        />
      </div>

      <PlotTooltip {...tooltip} />
    </div>
  );
};

export default LinePlot;
