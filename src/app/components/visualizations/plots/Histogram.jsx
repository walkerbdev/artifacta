import React, { useRef, useCallback } from 'react';
import { drawTopLegend, getChartFont } from '../../../hooks/useCanvasSetup';
import { useResponsiveCanvas } from '../../../hooks/useResponsiveCanvas';
import { getChartColor } from '../../../../core/utils/constants';

/**
 * Histogram Plot
 * Displays distribution of values with optional grouping
 */
const Histogram = ({ data, title: _title, metadata: _metadata }) => {
  const canvasRef = useRef(null);

  const drawHistogram = useCallback((width, height) => {
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

    // Compute histogram bins
    const { values, groups, bins = 30 } = data;

    // Draw legend FIRST if grouped and get actual height
    let legendHeight = 0;
    if (groups) {
      const groupNames = [...new Set(groups)];
      const legendItems = groupNames.map((group, i) => ({
        label: group,
        color: getChartColor(i)
      }));
      legendHeight = drawTopLegend(ctx, legendItems, width, 10, {
        leftMargin: 60,
        rightMargin: 40
      });
    }

    // Calculate padding with legend space
    const padding = {
      top: groups ? 10 + legendHeight + 10 : 40,
      right: 40,
      bottom: 100,
      left: 60
    };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    if (groups) {
      // Grouped histogram
      drawGroupedHistogram(ctx, values, groups, bins, padding, plotWidth, plotHeight, width, height);
    } else {
      // Simple histogram
      drawSimpleHistogram(ctx, values, bins, padding, plotWidth, plotHeight, width, height);
    }

    // Title is rendered by DraggableVisualization wrapper

  }, [data]);

  useResponsiveCanvas(canvasRef, drawHistogram);

  return (
    <div style={{ padding: '10px' }}>
      <canvas
        ref={canvasRef}
        className="viz-canvas-distribution"
        style={{ width: '100%', height: '100%', display: 'block' }}
      />
    </div>
  );
};

function drawSimpleHistogram(ctx, values, numBins, padding, plotWidth, plotHeight) {
  // Compute bins
  const min = Math.min(...values);
  const max = Math.max(...values);
  const binWidth = (max - min) / numBins;

  const bins = Array(numBins).fill(0);
  values.forEach(v => {
    const binIndex = Math.min(Math.floor((v - min) / binWidth), numBins - 1);
    bins[binIndex]++;
  });

  const maxCount = Math.max(...bins);

  // Draw bars
  const barWidth = plotWidth / numBins;
  ctx.fillStyle = getChartColor(0);

  bins.forEach((count, i) => {
    const barHeight = (count / maxCount) * plotHeight;
    const x = padding.left + i * barWidth;
    const y = padding.top + plotHeight - barHeight;

    ctx.fillRect(x, y, barWidth - 2, barHeight);
  });

  // Draw axes
  ctx.strokeStyle = '#333';
  ctx.lineWidth = 2;

  // Y-axis
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, padding.top + plotHeight);
  ctx.stroke();

  // X-axis
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top + plotHeight);
  ctx.lineTo(padding.left + plotWidth, padding.top + plotHeight);
  ctx.stroke();

  // Axis labels
  ctx.fillStyle = '#666';
  ctx.font = getChartFont('tick');
  ctx.textAlign = 'center';
  ctx.fillText(`Min: ${min.toFixed(2)}`, padding.left, padding.top + plotHeight + 30);
  ctx.fillText(`Max: ${max.toFixed(2)}`, padding.left + plotWidth, padding.top + plotHeight + 30);

  ctx.save();
  ctx.translate(20, padding.top + plotHeight / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.font = getChartFont('axisLabel');
  ctx.fillText('Count', 0, 0);
  ctx.restore();
}

function drawGroupedHistogram(ctx, values, groups, numBins, padding, plotWidth, plotHeight) {
  // Group values
  const groupedValues = {};
  values.forEach((v, i) => {
    const group = groups[i];
    if (!groupedValues[group]) groupedValues[group] = [];
    groupedValues[group].push(v);
  });

  const groupNames = Object.keys(groupedValues);

  // Legend already drawn in main function, skip here

  // Compute global min/max
  const min = Math.min(...values);
  const max = Math.max(...values);
  const binWidth = (max - min) / numBins;

  // Compute bins for each group
  const groupBins = {};
  let maxCount = 0;

  groupNames.forEach(group => {
    const bins = Array(numBins).fill(0);
    groupedValues[group].forEach(v => {
      const binIndex = Math.min(Math.floor((v - min) / binWidth), numBins - 1);
      bins[binIndex]++;
    });
    groupBins[group] = bins;
    maxCount = Math.max(maxCount, ...bins);
  });

  // Draw bars (stacked or side-by-side)
  const barWidth = plotWidth / numBins;
  const groupBarWidth = barWidth / groupNames.length;

  groupNames.forEach((group, groupIdx) => {
    ctx.fillStyle = getChartColor(groupIdx);

    groupBins[group].forEach((count, i) => {
      const barHeight = (count / maxCount) * plotHeight;
      const x = padding.left + i * barWidth + groupIdx * groupBarWidth;
      const y = padding.top + plotHeight - barHeight;

      ctx.fillRect(x, y, groupBarWidth - 2, barHeight);
    });
  });

  // Draw axes
  ctx.strokeStyle = '#333';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, padding.top + plotHeight);
  ctx.lineTo(padding.left + plotWidth, padding.top + plotHeight);
  ctx.stroke();

  // Draw x-axis tick labels (horizontal)
  const numXTicks = Math.min(10, groupNames.length);
  const baseY = padding.top + plotHeight + 15;

  ctx.fillStyle = '#666';
  ctx.font = getChartFont('tick');
  ctx.textAlign = 'center';

  for (let i = 0; i < numXTicks; i++) {
    const xVal = min + (i / (numXTicks - 1)) * (max - min);
    const x = padding.left + (i / (numXTicks - 1)) * plotWidth;
    ctx.fillText(xVal.toFixed(1), x, baseY);
  }

  // Axis label
  ctx.fillStyle = '#666';
  ctx.font = getChartFont('axisLabel');
  ctx.textAlign = 'center';
  ctx.fillText('Value', padding.left + plotWidth / 2, padding.top + plotHeight + 40);

  ctx.save();
  ctx.translate(20, padding.top + plotHeight / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.font = getChartFont('axisLabel');
  ctx.fillText('Count', 0, 0);
  ctx.restore();
}

export default Histogram;
