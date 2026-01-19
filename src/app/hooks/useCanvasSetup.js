import { useCallback } from 'react';

/**
 * Standardized font sizes across all chart types
 * Ensures consistent, human-readable text throughout the application
 */
export const CHART_FONTS = {
  // Legends
  legend: {
    size: 14,
    family: 'Arial',
    weight: 'normal'
  },
  // Axis labels (e.g., "Epoch", "Loss")
  axisLabel: {
    size: 14,
    family: 'Arial',
    weight: 'bold'
  },
  // Tick labels (numbers along axes)
  tick: {
    size: 12,
    family: 'Arial',
    weight: 'normal'
  },
  // Metric text (e.g., "AUC = 0.95")
  metric: {
    size: 14,
    family: 'sans-serif',
    weight: 'bold'
  },
  // Titles (when rendered in canvas)
  title: {
    size: 16,
    family: 'Arial',
    weight: 'bold'
  }
};

/**
 * Helper to get font string for canvas context
 * @param {'legend' | 'axisLabel' | 'tick' | 'metric' | 'title'} type - Font type
 * @returns {string} - CSS font string (e.g., "bold 14px Arial")
 */
export function getChartFont(type) {
  const font = CHART_FONTS[type];
  // Only include weight if it's bold (canvas doesn't need 'normal')
  const weight = font.weight === 'bold' ? 'bold ' : '';
  return `${weight}${font.size}px ${font.family}`;
}

/**
 * Reusable hook for setting up canvas with proper DPR scaling
 * Eliminates ~10 lines of boilerplate per chart component
 *
 * @param {React.RefObject} canvasRef - Reference to canvas element
 * @param {number} defaultWidth - Default width if clientWidth unavailable
 * @param {number} defaultHeight - Default height if clientHeight unavailable
 * @returns {Function} setupCanvas - Function that returns { ctx, width, height, dpr } or null
 */
export function useCanvasSetup(canvasRef, defaultWidth = 600, defaultHeight = 300) {
  const setupCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return null;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const displayWidth = canvas.clientWidth || defaultWidth;
    const displayHeight = canvas.clientHeight || defaultHeight;

    // Set actual canvas size (accounting for DPR for crisp rendering)
    canvas.width = displayWidth * dpr;
    canvas.height = displayHeight * dpr;

    // CRITICAL: Reset transform before scaling to prevent cumulative scaling
    // When React reuses canvas elements (same DOM node, different component),
    // the transform matrix persists. Without reset, each render compounds
    // the scale (e.g., scale(2,2) twice = 4x zoom instead of 2x)
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    // Scale context to match DPR
    ctx.scale(dpr, dpr);

    // Clear canvas with white background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, displayWidth, displayHeight);

    return {
      ctx,
      width: displayWidth,
      height: displayHeight,
      dpr
    };
  }, [canvasRef, defaultWidth, defaultHeight]);

  return setupCanvas;
}

/**
 * Standard padding configuration used across all charts
 */
const CHART_PADDING = {
  top: 20,
  right: 20,
  bottom: 40,
  left: 60
};

/**
 * Helper to get chart dimensions after padding
 */
export function getChartDimensions(width, height, padding = CHART_PADDING) {
  return {
    chartWidth: width - padding.left - padding.right,
    chartHeight: height - padding.top - padding.bottom
  };
}

/**
 * Draw axes on canvas
 */
export function drawAxes(ctx, width, height, padding, xLabel, yLabel) {
  ctx.strokeStyle = '#333';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, height - padding.bottom);
  ctx.lineTo(width - padding.right, height - padding.bottom);
  ctx.stroke();

  // X-axis label
  ctx.fillStyle = '#333';
  ctx.font = getChartFont('axisLabel');
  ctx.textAlign = 'center';
  ctx.fillText(xLabel, width / 2, height - 10);

  // Y-axis label
  ctx.save();
  ctx.translate(15, height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
}

/**
 * Draw grid lines with labels
 */
export function drawGridLines(ctx, width, height, padding, numLines = 5, formatLabel = (v) => v.toFixed(1)) {
  const { chartWidth, chartHeight } = getChartDimensions(width, height, padding);

  ctx.strokeStyle = '#e0e0e0';
  ctx.lineWidth = 1;

  for (let i = 0; i <= numLines; i++) {
    const val = i / numLines;
    const x = padding.left + (chartWidth * i) / numLines;
    const y = padding.top + (chartHeight * i) / numLines;

    // Vertical grid line
    ctx.beginPath();
    ctx.moveTo(x, padding.top);
    ctx.lineTo(x, height - padding.bottom);
    ctx.stroke();

    // Horizontal grid line
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();

    // X-axis label
    ctx.fillStyle = '#666';
    ctx.font = getChartFont('tick');
    ctx.textAlign = 'center';
    ctx.fillText(formatLabel(val), x, height - padding.bottom + 15);

    // Y-axis label
    ctx.textAlign = 'right';
    ctx.fillText(formatLabel(1 - val), padding.left - 5, y + 4);
  }
}

/**
 * Draw Y-axis with grid lines and labels
 * Eliminates ~30 lines of duplicate code per chart
 */
export function drawYAxisWithGrid(ctx, width, height, padding, yMin, yMax, formatYValue, numGridLines = 5) {
  ctx.strokeStyle = '#e0e0e0';
  ctx.lineWidth = 1;

  for (let i = 0; i <= numGridLines; i++) {
    const y = padding.top + (i / numGridLines) * (height - padding.top - padding.bottom);

    // Horizontal grid line
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();

    // Y-axis label
    const value = yMax - (i / numGridLines) * (yMax - yMin);
    ctx.fillStyle = '#666';
    ctx.font = getChartFont('tick');
    ctx.textAlign = 'right';
    const formattedValue = formatYValue(value);
    ctx.fillText(formattedValue, padding.left - 10, y + 4);
  }
}

/**
 * Render rotated X-axis labels with collision detection and staggering
 * Eliminates 40+ lines of duplicate code in LinePlot, Histogram, ViolinPlot
 *
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {Array<string>} labels - Array of label strings
 * @param {Array<number>} xPositions - X positions for each label
 * @param {number} baseY - Base Y position for labels
 * @param {Object} options - Optional configuration
 */
export function renderRotatedLabels(ctx, labels, xPositions, baseY, options = {}) {
  const {
    angle = -Math.PI / 4,
    color = '#666',
    buffer = 10,
    staggerOffset = 25
  } = options;

  const labelPositions = [];

  ctx.save();
  ctx.fillStyle = color;
  ctx.font = getChartFont('tick');
  ctx.textAlign = 'right';

  labels.forEach((label, i) => {
    const x = xPositions[i];
    let currentY = baseY;

    // Check for collisions with previous labels
    let hasCollision = true;
    let stagger = 0;

    while (hasCollision && stagger < 3) {
      hasCollision = false;
      const testY = baseY + (stagger * staggerOffset);

      for (const pos of labelPositions) {
        const distance = Math.sqrt(Math.pow(x - pos.x, 2) + Math.pow(testY - pos.y, 2));
        if (distance < buffer) {
          hasCollision = true;
          break;
        }
      }

      if (!hasCollision) {
        currentY = testY;
      } else {
        stagger++;
      }
    }

    // Render the label
    ctx.save();
    ctx.translate(x, currentY);
    ctx.rotate(angle);
    ctx.fillText(label, 0, 0);
    ctx.restore();

    labelPositions.push({ x, y: currentY });
  });

  ctx.restore();
}

/**
 * Draw legend at the top of the chart with automatic wrapping
 * Returns the height consumed by the legend for layout adjustment
 *
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {Array} items - Legend items [{label, color}, ...]
 * @param {number} width - Canvas width
 * @param {number} startY - Y position to start drawing legend (top of chart area)
 * @param {Object} options - Spacing and styling options
 * @returns {number} - Total height consumed by legend
 */
export function drawTopLegend(ctx, items, width, startY, options = {}) {
  const {
    leftMargin = 80,
    rightMargin = 20,
    lineWidth = 30,
    itemSpacing = 15,
    rowHeight = 24,
    textColor = '#333'
  } = options;

  ctx.font = getChartFont('legend');
  ctx.fillStyle = textColor;

  // First pass: calculate layout to determine row breaks and total width per row
  const rows = [[]];
  let currentRow = 0;
  let currentRowWidth = 0;
  const availableWidth = width - leftMargin - rightMargin;

  items.forEach((item) => {
    const textWidth = ctx.measureText(item.label).width;
    const itemWidth = lineWidth + 5 + textWidth + itemSpacing;

    // Check if item fits in current row
    if (currentRowWidth + itemWidth > availableWidth && rows[currentRow].length > 0) {
      // Start new row
      currentRow++;
      rows[currentRow] = [];
      currentRowWidth = 0;
    }

    rows[currentRow].push({ ...item, itemWidth });
    currentRowWidth += itemWidth;
  });

  // Second pass: draw each row centered
  let currentY = startY;
  rows.forEach((rowItems) => {
    // Calculate total width of this row
    const rowWidth = rowItems.reduce((sum, item) => sum + item.itemWidth, 0);

    // Center the row
    let currentX = leftMargin + (availableWidth - rowWidth) / 2;

    rowItems.forEach((item) => {
      // Draw legend line
      ctx.strokeStyle = item.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(currentX, currentY);
      ctx.lineTo(currentX + lineWidth, currentY);
      ctx.stroke();

      // Draw legend text
      ctx.fillStyle = textColor;
      ctx.textAlign = 'left';
      ctx.fillText(item.label, currentX + lineWidth + 5, currentY + 5);

      // Move to next position in row
      currentX += item.itemWidth;
    });

    currentY += rowHeight;
  });

  // Return total height consumed (rows * rowHeight + small buffer)
  return rows.length * rowHeight + 10;
}

/**
 * Calculate dynamic bottom padding for legend
 * Helps charts adjust height based on number of legend items
 */
export function calculateLegendPadding(numItems, canvasWidth, options = {}) {
  const {
    itemWidth = 150,
    rowHeight = 20,
    minPadding = 80,
    extraSpace = 50
  } = options;

  const availableWidth = canvasWidth - 100;
  const itemsPerRow = Math.max(1, Math.floor(availableWidth / itemWidth));
  const numRows = Math.ceil(numItems / itemsPerRow);
  const legendHeight = numRows * rowHeight + extraSpace;
  const result = Math.max(minPadding, legendHeight);

  return result;
}
