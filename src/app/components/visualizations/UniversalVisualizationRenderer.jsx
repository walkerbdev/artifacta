import React from 'react';
import LinePlot from './plots/LinePlot';
import Histogram from './plots/Histogram';
import ViolinPlot from './plots/ViolinPlot';
import Heatmap from './plots/Heatmap';
import CurveChart from './plots/CurveChart';
import ScatterPlot from './plots/ScatterPlot';
import BarChart from './plots/BarChart';

/**
 * Universal Visualization Renderer for plot type dispatch
 *
 * Routes plot configurations to the appropriate visualization component based on type.
 * Acts as a centralized dispatcher that maps primitive types from plot discovery
 * to their corresponding React components.
 *
 * Supported plot types:
 * - line: Line plots (time series, multi-series)
 * - scatter: Scatter plots (2D point clouds)
 * - heatmap: Heat maps (2D matrices, confusion matrices)
 * - barchart: Bar charts (categorical comparisons)
 * - histogram: Histograms (distributions)
 * - violin: Violin plots (distribution comparisons)
 * - curve: ROC/PR curves with metrics
 * - table: Data tables (handled separately in Tables tab)
 *
 * Architecture:
 * - Receives plotConfig from plot discovery system
 * - Type-based switch dispatch to appropriate component
 * - Forwards data, metadata, and title to child components
 * - Returns null for unsupported types or missing configs
 *
 * @param {object} props - Component props
 * @param {object} props.plotConfig - Plot configuration from discovery:
 *   - type: string - Plot type identifier
 *   - data: object - Plot-specific data structure
 *   - metadata: object (optional) - Additional plot metadata
 *   - title: string (optional) - Plot title
 * @returns {React.ReactElement|null} Rendered visualization component or error message
 */
const UniversalVisualizationRenderer = ({ plotConfig }) => {
  if (!plotConfig) return null;

  const { type, data, metadata, title } = plotConfig;

  switch (type) {
    case 'line':
      return <LinePlot data={data} title={title} metadata={metadata} />;

    case 'histogram':
      return <Histogram data={data} title={title} metadata={metadata} />;

    case 'violin':
      return <ViolinPlot data={data} title={title} metadata={metadata} />;

    case 'heatmap':
      return <Heatmap data={data} title={title} metadata={metadata} />;

    case 'table':
      // Table primitive is rendered in Tables tab, not as a visualization
      return null;

    case 'curve':
      // Check if data has curves array (multi-run) or is single curve
      if (data && data.curves) {
        return <CurveChart {...data} title={title} />;
      } else {
        return <CurveChart data={data} title={title} />;
      }

    case 'scatter':
      return <ScatterPlot data={data} title={title} metadata={metadata} />;

    case 'barchart':
      return <BarChart data={data} title={title} metadata={metadata} />;

    default:
      return (
        <div style={{ padding: '20px', textAlign: 'center', color: '#E94B3C' }}>
          Unsupported plot type: {type}
        </div>
      );
  }
};

export default UniversalVisualizationRenderer;
