import React, { useRef, useCallback } from 'react';
import { useCanvasTooltip } from '@/app/hooks/useCanvasTooltip';
import { useResponsiveCanvas } from '@/app/hooks/useResponsiveCanvas';
import { getChartFont } from '@/app/hooks/useCanvasSetup';
import PlotTooltip from '../shared/PlotTooltip';

/**
 * Heatmap component for 2D matrix visualization
 *
 * Renders color-coded grid heatmaps with interactive tooltips. Commonly used for
 * confusion matrices, correlation matrices, and attention weights.
 *
 * Features:
 * - Color-coded cells based on value (blue-white-red gradient)
 * - Interactive tooltips showing exact cell values
 * - Row and column labels
 * - Auto-scaled cell sizes based on matrix dimensions
 * - Value annotations in each cell
 * - HiDPI display support
 *
 * Data format:
 * ```
 * {
 *   rows: ["Class A", "Class B", "Class C"],      // Row labels
 *   cols: ["Pred A", "Pred B", "Pred C"],        // Column labels
 *   values: [                                      // 2D matrix
 *     [120, 5, 2],    // Row 0
 *     [3, 95, 8],     // Row 1
 *     [1, 10, 110]    // Row 2
 *   ]
 * }
 * ```
 *
 * @param {object} props - Component props
 * @param {object} props.data - Heatmap data with labels and values
 * @param {Array<string>} props.data.rows - Row labels
 * @param {Array<string>} props.data.cols - Column labels
 * @param {Array<Array<number>>} props.data.values - 2D matrix of numeric values
 * @param {string} [props._title] - Title (handled by wrapper, not used internally)
 * @returns {React.ReactElement} Canvas-based heatmap with tooltip
 */
const Heatmap = ({ data, _title }) => {
  const canvasRef = useRef(null);
  const plotDataRef = useRef(null);

  /**
   * Draws the heatmap on canvas
   * @param {number} width - Canvas width
   * @param {number} height - Canvas height
   * @returns {void}
   */
  const drawHeatmap = useCallback((width, height) => {
    if (!data || !data.values || !canvasRef.current) {
      return;
    }

    const canvas = canvasRef.current;
    const dpr = window.devicePixelRatio || 1;

    const { rows, cols, values } = data;

    // Set canvas bitmap size - only update if changed to prevent loops
    const targetWidth = Math.floor(width * dpr);
    const targetHeight = Math.floor(height * dpr);

    if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
      canvas.width = targetWidth;
      canvas.height = targetHeight;
    }

    const ctx = canvas.getContext('2d');
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);

    // Use fixed cell size for confusion matrices - they should NOT scale with container
    // This keeps matrix stable across multi-run selections
    const cellSize = 80;

    const chartWidth = cellSize * cols.length;
    const chartHeight = cellSize * rows.length;

    // Center chart both horizontally and vertically in canvas
    const chartStartX = (width - chartWidth) / 2;
    const chartStartY = (height - chartHeight) / 2;

    // Adjust padding to position labels around centered chart
    const padding = {
      top: chartStartY,
      right: width - chartStartX - chartWidth,
      bottom: height - chartStartY - chartHeight,
      left: chartStartX
    };

    // Clear canvas
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    // Find min/max for color scaling
    const allValues = values.flat();
    const minVal = Math.min(...allValues);
    const maxVal = Math.max(...allValues);

    // Color scale: blue (low) -> white (mid) -> red (high)
    /**
     * Calculates color for a value based on min/max range
     * @param {number} value - The value to calculate color for
     * @returns {string} RGB color string
     */
    const getColor = (value) => {
      const normalized = (value - minVal) / (maxVal - minVal);

      if (normalized < 0.5) {
        // Blue to white
        const t = normalized * 2;
        const r = Math.round(74 + (255 - 74) * t);
        const g = Math.round(144 + (255 - 144) * t);
        const b = Math.round(226 + (255 - 226) * t);
        return `rgb(${r}, ${g}, ${b})`;
      } else {
        // White to red
        const t = (normalized - 0.5) * 2;
        const r = 255;
        const g = Math.round(255 - 255 * t);
        const b = Math.round(255 - 255 * t);
        return `rgb(${r}, ${g}, ${b})`;
      }
    };

    // Draw cells
    values.forEach((row, i) => {
      row.forEach((value, j) => {
        const x = chartStartX + j * cellSize;
        const y = padding.top + i * cellSize;

        // Fill cell
        ctx.fillStyle = getColor(value);
        ctx.fillRect(x, y, cellSize, cellSize);

        // Draw cell border
        ctx.strokeStyle = '#ddd';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, cellSize, cellSize);

        // Draw value with dynamic font size
        ctx.fillStyle = '#333';
        const fontSize = Math.max(8, Math.min(14, Math.floor(cellSize / 2.5), Math.floor(cellSize * 0.6)));
        ctx.font = `${fontSize}px Arial`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(value.toFixed(0), x + cellSize / 2, y + cellSize / 2);
      });
    });

    // Draw row labels
    ctx.fillStyle = '#333';
    ctx.font = getChartFont('tick');
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    rows.forEach((label, i) => {
      const y = padding.top + i * cellSize + cellSize / 2;
      ctx.fillText(label, chartStartX - 10, y);
    });

    // Draw column labels (rotated 45°)
    ctx.font = getChartFont('axisLabel');
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    cols.forEach((label, j) => {
      const x = chartStartX + j * cellSize + cellSize / 2;
      ctx.save();
      ctx.translate(x, padding.top - 5);
      ctx.rotate(-Math.PI / 4);
      ctx.fillText(label, -3, 0);
      ctx.restore();
    });

    // Title is rendered by DraggableVisualization wrapper

    // Draw color scale legend on right side
    const legendWidth = 20;
    const legendHeight = chartHeight;
    const legendX = width - padding.right + 20;
    const legendY = padding.top;

    // Draw gradient (vertical)
    for (let i = 0; i < legendHeight; i++) {
      const normalized = 1 - (i / legendHeight); // Inverted: high at top, low at bottom
      ctx.fillStyle = getColor(minVal + normalized * (maxVal - minVal));
      ctx.fillRect(legendX, legendY + i, legendWidth, 1);
    }

    // Draw legend border
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    ctx.strokeRect(legendX, legendY, legendWidth, legendHeight);

    // Draw legend labels
    ctx.fillStyle = '#333';
    ctx.font = getChartFont('tick');
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(maxVal.toFixed(1), legendX + legendWidth + 5, legendY);
    ctx.fillText(minVal.toFixed(1), legendX + legendWidth + 5, legendY + legendHeight);

    // Store data for tooltip (on-the-fly cell detection)
    plotDataRef.current = {
      rows,
      cols,
      values,
      padding,
      cellSize,
      chartStartX
    };

    // No return needed - container controls size, plot adapts

  }, [data]);

  useResponsiveCanvas(canvasRef, drawHeatmap);

  /**
   * Tooltip logic: find cell under mouse (on-the-fly)
   * @param {number} mouseX - Mouse X coordinate
   * @param {number} mouseY - Mouse Y coordinate
   * @returns {object|null} Tooltip data or null if not over a cell
   */
  const getTooltipData = useCallback((mouseX, mouseY) => {
    if (!plotDataRef.current) return null;

    const { rows, cols, values, padding, cellSize, chartStartX } = plotDataRef.current;

    // Check if mouse is within grid bounds
    const chartX = mouseX - chartStartX;
    const chartY = mouseY - padding.top;

    if (chartX < 0 || chartY < 0) return null;

    const colIndex = Math.floor(chartX / cellSize);
    const rowIndex = Math.floor(chartY / cellSize);

    if (colIndex < 0 || colIndex >= cols.length || rowIndex < 0 || rowIndex >= rows.length) {
      return null;
    }

    const value = values[rowIndex][colIndex];

    return {
      type: 'matrix',
      content: {
        row: rows[rowIndex],
        col: cols[colIndex],
        value
      }
    };
  }, []);

  const tooltip = useCanvasTooltip({
    canvasRef,
    getTooltipData,
    searchRadius: 1000 // Large radius since we're doing grid detection, not distance
  });

  // Calculate fixed canvas size based on data dimensions
  const canvasWidth = data ? 600 : 500;
  const canvasHeight = data ? 500 : 500;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', width: '100%', height: '100%' }}>
      <div style={{ flex: 1, minHeight: 0, padding: '10px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <canvas
          ref={canvasRef}
          className="viz-canvas-confusion"
          style={{ width: `${canvasWidth}px`, height: `${canvasHeight}px`, display: 'block' }}
        />
      </div>
      <PlotTooltip {...tooltip} />
    </div>
  );
};

export default Heatmap;
