import React from 'react';
import './PlotTooltip.css';

/**
 * Unified tooltip component for all plot types
 * Handles positioning, formatting, and rendering of tooltip data
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
