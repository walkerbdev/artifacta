import React from 'react';
import './PlotTooltip.css';

/**
 * Plot Tooltip component for displaying data point details on hover
 *
 * Unified tooltip rendering for all plot types (line, scatter, heatmap, curves, etc.).
 * Automatically positions itself to avoid screen edges and formats data appropriately.
 *
 * Features:
 * - Smart edge detection (flips to left/top/bottom if near edge)
 * - Type-specific rendering (series, scatter, matrix, curve, distribution)
 * - Custom value formatters
 * - Fixed positioning (follows cursor)
 * - High z-index (always on top)
 * - Pointer-events: none (doesn't block mouse)
 *
 * Supported data types:
 * - series: Time series with index + multiple values
 * - scatter: X/Y coordinates with optional label
 * - matrix: Row/col/value for heatmaps
 * - curve: X/Y for ROC/PR curves with metric display
 * - distribution: Count + range for histograms
 * - generic: Key-value pairs for fallback
 *
 * @param {object} props - Component props
 * @param {boolean} props.visible - Whether tooltip should be shown
 * @param {number} props.x - Screen X coordinate (from mouse event)
 * @param {number} props.y - Screen Y coordinate (from mouse event)
 * @param {object|null} props.data - Tooltip data: { type, content }
 * @param {object} [props.formatters={}] - Custom formatters:
 *   - value: (v) => string - Format numeric values
 *   - index: (v) => string - Format index values
 * @returns {React.ReactElement|null} Positioned tooltip or null if not visible
 */
const PlotTooltip = ({ visible, x, y, data, formatters = {} }) => {
  if (!visible || !data) return null;

  // Auto-position to avoid edges
  const tooltipStyle = {
    position: 'fixed',
    left: `${x}px`,
    top: `${y}px`,
    transform: 'translate(10px, -50%)', // Default: right of cursor
    pointerEvents: 'none',
    zIndex: 9999
  };

  // Adjust if too close to right edge
  if (x > window.innerWidth - 250) {
    tooltipStyle.transform = 'translate(calc(-100% - 10px), -50%)'; // Left of cursor
  }

  // Adjust if too close to top/bottom
  if (y < 100) {
    tooltipStyle.transform = tooltipStyle.transform.replace('-50%', '10px');
  } else if (y > window.innerHeight - 100) {
    tooltipStyle.transform = tooltipStyle.transform.replace('-50%', 'calc(-100% - 10px)');
  }

  return (
    <div className="plot-tooltip" style={tooltipStyle}>
      <div className="plot-tooltip__content">
        {renderTooltipContent(data, formatters)}
      </div>
    </div>
  );
};

/**
 * Render tooltip content based on data structure
 * @param {object} data - Data object containing type and content
 * @param {object} formatters - Formatters for value display
 * @returns {object} Rendered tooltip content
 */
function renderTooltipContent(data, formatters) {
  const { type, content } = data;

  switch (type) {
    case 'series':
      return renderSeriesData(content, formatters);
    case 'scatter':
      return renderScatterData(content, formatters);
    case 'matrix':
      return renderMatrixData(content, formatters);
    case 'curve':
      return renderCurveData(content, formatters);
    case 'distribution':
      return renderDistributionData(content, formatters);
    default:
      return renderGenericData(content, formatters);
  }
}

/**
 * Render series data tooltip
 * @param {object} content - Series content with index and values
 * @param {object} formatters - Formatters for value display
 * @returns {object} Rendered series tooltip
 */
function renderSeriesData(content, formatters) {
  const { index, indexLabel, values } = content;
  const formatValue = formatters.value || ((v) => typeof v === 'number' ? v.toFixed(4) : v);
  const formatIndex = formatters.index || ((v) => v);

  return (
    <>
      <div className="plot-tooltip__header">
        {indexLabel || 'Index'}: {formatIndex(index)}
      </div>
      <div className="plot-tooltip__body">
        {Object.entries(values).map(([key, value]) => (
          <div key={key} className="plot-tooltip__row">
            <span className="plot-tooltip__label">{key}:</span>
            <span className="plot-tooltip__value">{formatValue(value)}</span>
          </div>
        ))}
      </div>
    </>
  );
}

/**
 * Render scatter plot data tooltip
 * @param {object} content - Scatter content with x, y coordinates and label
 * @param {object} formatters - Formatters for value display
 * @returns {object} Rendered scatter tooltip
 */
function renderScatterData(content, formatters) {
  const { x, y, label, color } = content;
  const formatValue = formatters.value || ((v) => typeof v === 'number' ? v.toFixed(4) : v);

  return (
    <>
      {label && <div className="plot-tooltip__header">{label}</div>}
      <div className="plot-tooltip__body">
        <div className="plot-tooltip__row">
          <span className="plot-tooltip__label">x:</span>
          <span className="plot-tooltip__value">{formatValue(x)}</span>
        </div>
        <div className="plot-tooltip__row">
          <span className="plot-tooltip__label">y:</span>
          <span className="plot-tooltip__value">{formatValue(y)}</span>
        </div>
        {color && (
          <div className="plot-tooltip__row">
            <span
              className="plot-tooltip__color-indicator"
              style={{ backgroundColor: color }}
            />
          </div>
        )}
      </div>
    </>
  );
}

/**
 * Render matrix data tooltip
 * @param {object} content - Matrix content with row, column and value
 * @param {object} formatters - Formatters for value display
 * @returns {object} Rendered matrix tooltip
 */
function renderMatrixData(content, formatters) {
  const { row, col, value } = content;
  const formatValue = formatters.value || ((v) => typeof v === 'number' ? v.toFixed(2) : v);

  return (
    <div className="plot-tooltip__body">
      <div className="plot-tooltip__row">
        <span className="plot-tooltip__label">Row:</span>
        <span className="plot-tooltip__value">{row}</span>
      </div>
      <div className="plot-tooltip__row">
        <span className="plot-tooltip__label">Col:</span>
        <span className="plot-tooltip__value">{col}</span>
      </div>
      <div className="plot-tooltip__row">
        <span className="plot-tooltip__label">Value:</span>
        <span className="plot-tooltip__value">{formatValue(value)}</span>
      </div>
    </div>
  );
}

/**
 * Render curve data tooltip
 * @param {object} content - Curve content with x, y coordinates and labels
 * @param {object} formatters - Formatters for value display
 * @returns {object} Rendered curve tooltip
 */
function renderCurveData(content, formatters) {
  const { x, y, xLabel, yLabel, metric } = content;
  const formatValue = formatters.value || ((v) => typeof v === 'number' ? v.toFixed(4) : v);

  return (
    <>
      {metric && <div className="plot-tooltip__header">{metric}</div>}
      <div className="plot-tooltip__body">
        <div className="plot-tooltip__row">
          <span className="plot-tooltip__label">{xLabel || 'x'}:</span>
          <span className="plot-tooltip__value">{formatValue(x)}</span>
        </div>
        <div className="plot-tooltip__row">
          <span className="plot-tooltip__label">{yLabel || 'y'}:</span>
          <span className="plot-tooltip__value">{formatValue(y)}</span>
        </div>
      </div>
    </>
  );
}

/**
 * Render distribution data tooltip
 * @param {object} content - Distribution content with count and range
 * @param {object} formatters - Formatters for value display
 * @returns {object} Rendered distribution tooltip
 */
function renderDistributionData(content, formatters) {
  const { count, range } = content;
  const formatValue = formatters.value || ((v) => typeof v === 'number' ? v.toFixed(2) : v);

  return (
    <div className="plot-tooltip__body">
      {range && (
        <div className="plot-tooltip__row">
          <span className="plot-tooltip__label">Range:</span>
          <span className="plot-tooltip__value">
            {formatValue(range[0])} - {formatValue(range[1])}
          </span>
        </div>
      )}
      <div className="plot-tooltip__row">
        <span className="plot-tooltip__label">Count:</span>
        <span className="plot-tooltip__value">{count}</span>
      </div>
    </div>
  );
}

/**
 * Render generic data tooltip
 * @param {object} content - Generic content object with key-value pairs
 * @param {object} formatters - Formatters for value display
 * @returns {object} Rendered generic tooltip
 */
function renderGenericData(content, formatters) {
  const formatValue = formatters.value || ((v) => typeof v === 'number' ? v.toFixed(4) : v);

  return (
    <div className="plot-tooltip__body">
      {Object.entries(content).map(([key, value]) => (
        <div key={key} className="plot-tooltip__row">
          <span className="plot-tooltip__label">{key}:</span>
          <span className="plot-tooltip__value">{formatValue(value)}</span>
        </div>
      ))}
    </div>
  );
}

export default PlotTooltip;
