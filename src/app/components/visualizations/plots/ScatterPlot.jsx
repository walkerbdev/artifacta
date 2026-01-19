import React, { useRef, useCallback } from 'react';
import { useCanvasTooltip } from '@/app/hooks/useCanvasTooltip';
import { useResponsiveCanvas } from '@/app/hooks/useResponsiveCanvas';
import { getChartFont } from '@/app/hooks/useCanvasSetup';
import { CHART_PADDING } from '@/core/utils/constants';
import PlotTooltip from '../shared/PlotTooltip';

/**
 * Scatter Plot for Scatter primitive
 *
 * Expected data format (scatter primitive):
 * {
 *   points: [{<field1>: val, <field2>: val, label?, size?, color?}, ...],
 *   x_label: "X Axis Label",
 *   y_label: "Y Axis Label"
 * }
 *
 * Auto-detects numeric fields - does NOT assume field names.
 * First two numeric fields are used as x and y.
 */
const ScatterPlot = ({ data }) => {
  const canvasRef = useRef(null);
  const plotDataRef = useRef(null);

  const drawScatter = useCallback((width, height) => {
    if (!data || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    // Handle transformed format: {x: Array, y: Array, xLabel, yLabel}
    const { x: xArray, y: yArray, xLabel = 'X', yLabel = 'Y' } = data;

    if (!xArray || !yArray || xArray.length === 0 || yArray.length === 0) return;

    // Set canvas size
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);

    const padding = CHART_PADDING;
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // Clear canvas
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    // Filter out null/NaN values
    const xValues = xArray.filter(v => v != null && !isNaN(v));
    const yValues = yArray.filter(v => v != null && !isNaN(v));

    if (xValues.length === 0 || yValues.length === 0) return;

    // Find ranges
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);

    // Add 10% padding
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;
    const xPadded = xRange * 0.1;
    const yPadded = yRange * 0.1;

    const xAxisMin = xMin - xPadded;
    const xAxisMax = xMax + xPadded;
    const yAxisMin = yMin - yPadded;
    const yAxisMax = yMax + yPadded;

    // Draw axes
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, height - padding.bottom);
    ctx.lineTo(width - padding.right, height - padding.bottom);
    ctx.stroke();

    // Draw grid lines
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;
    const numGridLines = 5;

    for (let i = 0; i <= numGridLines; i++) {
      // Vertical grid lines
      const x = padding.left + (i / numGridLines) * chartWidth;
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, height - padding.bottom);
      ctx.stroke();

      // X-axis labels
      const xValue = xAxisMin + (i / numGridLines) * (xAxisMax - xAxisMin);
      ctx.fillStyle = '#666';
      ctx.font = getChartFont('tick');
      ctx.textAlign = 'center';
      ctx.fillText(xValue.toFixed(2), x, height - padding.bottom + 20);

      // Horizontal grid lines
      const y = padding.top + (i / numGridLines) * chartHeight;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();

      // Y-axis labels
      const yValue = yAxisMax - (i / numGridLines) * (yAxisMax - yAxisMin);
      ctx.fillStyle = '#666';
      ctx.font = getChartFont('tick');
      ctx.textAlign = 'right';
      ctx.fillText(yValue.toFixed(2), padding.left - 5, y + 4);
    }

    // Axis labels
    ctx.fillStyle = '#333';
    ctx.font = getChartFont('axisLabel');
    ctx.textAlign = 'center';
    ctx.fillText(xLabel, width / 2, height - 10);

    ctx.save();
    ctx.translate(20, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();

    // Draw points from x and y arrays
    for (let i = 0; i < Math.min(xArray.length, yArray.length); i++) {
      const xVal = xArray[i];
      const yVal = yArray[i];

      if (xVal == null || yVal == null || isNaN(xVal) || isNaN(yVal)) continue;

      const canvasX = padding.left + ((xVal - xAxisMin) / (xAxisMax - xAxisMin)) * chartWidth;
      const canvasY = padding.top + chartHeight - ((yVal - yAxisMin) / (yAxisMax - yAxisMin)) * chartHeight;

      ctx.fillStyle = '#4A90E2';
      ctx.globalAlpha = 0.6;
      ctx.beginPath();
      ctx.arc(canvasX, canvasY, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1.0;
    }

    // Store data for tooltip (on-the-fly transform)
    plotDataRef.current = {
      xArray,
      yArray,
      padding,
      xAxisMin,
      xAxisMax,
      yAxisMin,
      yAxisMax,
      chartWidth,
      chartHeight,
      xLabel,
      yLabel
    };

  }, [data]);

  useResponsiveCanvas(canvasRef, drawScatter);

  // Tooltip logic: find nearest point (on-the-fly transform)
  const getTooltipData = useCallback((mouseX, mouseY, searchRadius) => {
    if (!plotDataRef.current) return null;

    const pd = plotDataRef.current;
    const { xArray, yArray, padding, xAxisMin, xAxisMax, yAxisMin, yAxisMax, chartWidth, chartHeight, xLabel, yLabel } = pd;

    // Find nearest point within search radius
    let nearestPoint = null;
    let minDistance = searchRadius;
    let nearestIndex = -1;

    for (let i = 0; i < Math.min(xArray.length, yArray.length); i++) {
      const xVal = xArray[i];
      const yVal = yArray[i];

      if (xVal == null || yVal == null || isNaN(xVal) || isNaN(yVal)) continue;

      // Transform to canvas coordinates on-the-fly
      const canvasX = padding.left + ((xVal - xAxisMin) / (xAxisMax - xAxisMin)) * chartWidth;
      const canvasY = padding.top + chartHeight - ((yVal - yAxisMin) / (yAxisMax - yAxisMin)) * chartHeight;

      const distance = Math.sqrt(
        Math.pow(canvasX - mouseX, 2) +
        Math.pow(canvasY - mouseY, 2)
      );

      if (distance < minDistance) {
        minDistance = distance;
        nearestPoint = { x: xVal, y: yVal };
        nearestIndex = i;
      }
    }

    if (!nearestPoint) return null;

    return {
      type: 'scatter',
      content: {
        x: nearestPoint.x,
        y: nearestPoint.y,
        xLabel,
        yLabel,
        label: `Point ${nearestIndex}`
      }
    };
  }, []);

  const tooltip = useCanvasTooltip({
    canvasRef,
    getTooltipData,
    searchRadius: 15
  });

  return (
    <div style={{ padding: '10px' }}>
      <canvas
        ref={canvasRef}
        className="viz-canvas-scatter"
        style={{ width: '100%', height: 'auto', display: 'block' }}
      />
      <PlotTooltip {...tooltip} />
    </div>
  );
};

export default ScatterPlot;
