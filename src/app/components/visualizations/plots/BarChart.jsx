import React, { useRef, useCallback } from 'react';
import { renderRotatedLabels, drawTopLegend, getChartFont } from '../../../hooks/useCanvasSetup';
import { useResponsiveCanvas } from '../../../hooks/useResponsiveCanvas';
import { getChartColor } from '../../../../core/utils/constants';

/**
 * Bar Chart
 * Displays categorical data with grouped or stacked bars
 */
const BarChart = ({ data, title: _title, metadata: _metadata }) => {
  const canvasRef = useRef(null);

  const drawBarChart = useCallback((width, height) => {
    if (!data) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    // Setup canvas with DPR
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);

    // Clear canvas
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    const { categories, groups, x_label, y_label, stacked = false } = data;

    if (!categories || !groups) return;

    const groupNames = Object.keys(groups);
    const numGroups = groupNames.length;

    // Draw legend FIRST if multiple groups
    let legendHeight = 0;
    if (numGroups > 1) {
      const legendItems = groupNames.map((groupName, i) => ({
        label: groupName,
        color: getChartColor(i)
      }));
      legendHeight = drawTopLegend(ctx, legendItems, width, 10, {
        leftMargin: 80,
        rightMargin: 40
      });
    }

    // Calculate padding
    const padding = {
      top: numGroups > 1 ? 10 + legendHeight + 10 : 40,
      right: 40,
      bottom: 60,
      left: 80
    };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    // Find min/max values for y-axis
    let minValue = 0;
    let maxValue = 0;

    if (stacked) {
      // For stacked, sum values across groups for each category
      categories.forEach((_, catIdx) => {
        let sum = 0;
        groupNames.forEach(groupName => {
          sum += groups[groupName][catIdx] || 0;
        });
        maxValue = Math.max(maxValue, sum);
      });
    } else {
      // For grouped, find overall min/max
      groupNames.forEach(groupName => {
        const values = groups[groupName];
        minValue = Math.min(minValue, ...values);
        maxValue = Math.max(maxValue, ...values);
      });
    }

    // Add 10% padding to max
    const range = maxValue - minValue;
    maxValue = maxValue + range * 0.1;
    if (minValue < 0) {
      minValue = minValue - range * 0.1;
    }

    // Draw bars (horizontal mode not yet implemented)
    drawVerticalBars(ctx, categories, groups, groupNames, stacked, padding, plotWidth, plotHeight, width, height, minValue, maxValue, x_label, y_label);

  }, [data]);

  useResponsiveCanvas(canvasRef, drawBarChart);

  return (
    <div style={{ padding: '10px' }}>
      <canvas
        ref={canvasRef}
        className="viz-canvas-barchart"
        style={{ width: '100%', height: '100%', display: 'block' }}
      />
    </div>
  );
};

function drawVerticalBars(ctx, categories, groups, groupNames, stacked, padding, plotWidth, plotHeight, canvasWidth, canvasHeight, minValue, maxValue, xLabel, yLabel) {
  const numCategories = categories.length;
  const numGroups = groupNames.length;

  // Draw axes
  ctx.strokeStyle = '#333';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, padding.top + plotHeight);
  ctx.lineTo(padding.left + plotWidth, padding.top + plotHeight);
  ctx.stroke();

  // Y-axis labels
  const numYTicks = 5;
  ctx.fillStyle = '#666';
  ctx.font = getChartFont('tick');
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';

  for (let i = 0; i <= numYTicks; i++) {
    const value = minValue + (maxValue - minValue) * (i / numYTicks);
    const y = padding.top + plotHeight - (i / numYTicks) * plotHeight;

    ctx.fillText(value.toFixed(2), padding.left - 10, y);

    // Grid line
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(padding.left + plotWidth, y);
    ctx.stroke();
  }

  // Y-axis label
  if (yLabel) {
    ctx.save();
    ctx.translate(20, padding.top + plotHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.font = getChartFont('axisLabel');
    ctx.fillStyle = '#333';
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
  }

  // Calculate bar dimensions
  const categoryWidth = plotWidth / numCategories;
  const barPadding = categoryWidth * 0.2;
  const availableWidth = categoryWidth - barPadding;
  const barWidth = stacked ? availableWidth * 0.6 : availableWidth / numGroups;

  // Draw bars
  categories.forEach((category, catIdx) => {
    const categoryX = padding.left + catIdx * categoryWidth;

    if (stacked) {
      // Stacked bars
      let stackY = padding.top + plotHeight;
      groupNames.forEach((groupName, groupIdx) => {
        const value = groups[groupName][catIdx] || 0;
        const barHeight = (value / (maxValue - minValue)) * plotHeight;

        ctx.fillStyle = getChartColor(groupIdx);
        const barX = categoryX + barPadding / 2 + (availableWidth - barWidth) / 2;
        ctx.fillRect(barX, stackY - barHeight, barWidth, barHeight);

        stackY -= barHeight;
      });
    } else {
      // Grouped bars
      groupNames.forEach((groupName, groupIdx) => {
        const value = groups[groupName][catIdx] || 0;
        const normalizedValue = (value - minValue) / (maxValue - minValue);
        const barHeight = normalizedValue * plotHeight;

        ctx.fillStyle = getChartColor(groupIdx);
        const barX = categoryX + barPadding / 2 + groupIdx * barWidth;
        const barY = padding.top + plotHeight - barHeight;
        ctx.fillRect(barX, barY, barWidth, barHeight);
      });
    }
  });

  // X-axis category labels
  renderRotatedLabels(
    ctx,
    categories,
    padding.left,
    padding.top + plotHeight + 10,
    plotWidth,
    45,
    12
  );

  // X-axis label
  if (xLabel) {
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.font = getChartFont('axisLabel');
    ctx.fillStyle = '#333';
    ctx.fillText(xLabel, padding.left + plotWidth / 2, canvasHeight - 20);
  }
}

export default BarChart;
