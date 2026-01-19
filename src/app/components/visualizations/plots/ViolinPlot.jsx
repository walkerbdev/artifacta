import React, { useRef, useCallback } from 'react';
import { getChartFont } from '@/app/hooks/useCanvasSetup';
import { useResponsiveCanvas } from '@/app/hooks/useResponsiveCanvas';
import { getChartColor, CHART_PADDING } from '../../../../core/utils/constants';

/**
 * Violin Plot - Shows distribution density with embedded box plot
 * Displays distribution of values across groups using kernel density estimation
 */
const ViolinPlot = ({ data, title: _title, metadata: _metadata }) => {
  const canvasRef = useRef(null);

  const drawViolin = useCallback((width, height) => {
    if (!data || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    const padding = CHART_PADDING;
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    const { groups, data: groupData } = data;

    if (!groups || !groupData) return;

    // Compute statistics and density for each group
    const stats = groupData.map(values => computeStats(values));
    const densities = groupData.map(values => computeKDE(values));

    // Find global min/max for y-axis
    const allValues = groupData.flat();
    const yMin = Math.min(...allValues);
    const yMax = Math.max(...allValues);
    const yRange = yMax - yMin;

    const groupWidth = plotWidth / groups.length;
    const violinWidth = Math.min(80, groupWidth * 0.7);

    // Draw violins
    groups.forEach((group, i) => {
      const stat = stats[i];
      const density = densities[i];
      const x = padding.left + i * groupWidth + groupWidth / 2;
      const color = getChartColor(i);

      // Draw violin shape (mirrored density curve)
      ctx.fillStyle = color.replace('rgb', 'rgba').replace(')', ', 0.3)');
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;

      ctx.beginPath();

      // Right side of violin (going up)
      density.points.forEach((point, idx) => {
        const y = padding.top + plotHeight - ((point.value - yMin) / yRange) * plotHeight;
        const densityWidth = (point.density / density.maxDensity) * (violinWidth / 2);

        if (idx === 0) {
          ctx.moveTo(x + densityWidth, y);
        } else {
          ctx.lineTo(x + densityWidth, y);
        }
      });

      // Left side of violin (going down)
      for (let idx = density.points.length - 1; idx >= 0; idx--) {
        const point = density.points[idx];
        const y = padding.top + plotHeight - ((point.value - yMin) / yRange) * plotHeight;
        const densityWidth = (point.density / density.maxDensity) * (violinWidth / 2);
        ctx.lineTo(x - densityWidth, y);
      }

      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // Draw mini box plot inside violin
      const boxWidth = violinWidth * 0.15;
      const q1Y = padding.top + plotHeight - ((stat.q1 - yMin) / yRange) * plotHeight;
      const q3Y = padding.top + plotHeight - ((stat.q3 - yMin) / yRange) * plotHeight;
      const medianY = padding.top + plotHeight - ((stat.median - yMin) / yRange) * plotHeight;

      // Box (Q1 to Q3)
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(x - boxWidth / 2, q3Y, boxWidth, q1Y - q3Y);

      // Median line
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x - boxWidth / 2, medianY);
      ctx.lineTo(x + boxWidth / 2, medianY);
      ctx.stroke();

      // Whiskers
      const minY = padding.top + plotHeight - ((stat.min - yMin) / yRange) * plotHeight;
      const maxY = padding.top + plotHeight - ((stat.max - yMin) / yRange) * plotHeight;

      ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)';
      ctx.lineWidth = 1;

      // Lower whisker
      ctx.beginPath();
      ctx.moveTo(x, q1Y);
      ctx.lineTo(x, minY);
      ctx.stroke();

      // Upper whisker
      ctx.beginPath();
      ctx.moveTo(x, q3Y);
      ctx.lineTo(x, maxY);
      ctx.stroke();
    });

    // Draw group labels (horizontal, no rotation)
    ctx.fillStyle = '#333';
    ctx.font = getChartFont('tick');
    ctx.textAlign = 'center';

    const baseY = padding.top + plotHeight + 15;

    groups.forEach((group, i) => {
      const x = padding.left + i * groupWidth + groupWidth / 2;
      ctx.fillText(group, x, baseY);
    });

    // Draw axes
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + plotHeight);
    ctx.lineTo(padding.left + plotWidth, padding.top + plotHeight);
    ctx.stroke();

    // Y-axis labels
    ctx.fillStyle = '#666';
    ctx.font = getChartFont('tick');
    ctx.textAlign = 'right';
    [yMin, (yMin + yMax) / 2, yMax].forEach((value, i) => {
      const y = padding.top + plotHeight - (i / 2) * plotHeight;
      ctx.fillText(value.toFixed(2), padding.left - 10, y + 5);
    });

  }, [data]);

  useResponsiveCanvas(canvasRef, drawViolin);

  return (
    <div style={{ padding: '10px' }}>
      <canvas
        ref={canvasRef}
        className="viz-canvas-distribution"
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
};

/**
 * Compute kernel density estimation for violin plot
 */
function computeKDE(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  const min = sorted[0];
  const max = sorted[n - 1];
  const range = max - min;

  // Bandwidth using Silverman's rule of thumb
  const std = Math.sqrt(values.reduce((sum, v) => sum + Math.pow(v - (values.reduce((a, b) => a + b, 0) / n), 2), 0) / n);
  const bandwidth = 1.06 * std * Math.pow(n, -1/5);

  // Sample points along the range
  const numPoints = 50;
  const points = [];
  let maxDensity = 0;

  for (let i = 0; i <= numPoints; i++) {
    const value = min + (i / numPoints) * range;

    // Gaussian kernel density estimation
    let density = 0;
    for (const v of values) {
      const u = (value - v) / bandwidth;
      density += Math.exp(-0.5 * u * u) / Math.sqrt(2 * Math.PI);
    }
    density = density / (n * bandwidth);

    maxDensity = Math.max(maxDensity, density);
    points.push({ value, density });
  }

  return { points, maxDensity };
}

function computeStats(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;

  return {
    min: sorted[0],
    max: sorted[n - 1],
    median: sorted[Math.floor(n / 2)],
    q1: sorted[Math.floor(n / 4)],
    q3: sorted[Math.floor(3 * n / 4)]
  };
}

export default ViolinPlot;
