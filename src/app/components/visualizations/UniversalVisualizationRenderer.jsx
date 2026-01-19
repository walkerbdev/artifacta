import React from 'react';
import LinePlot from './plots/LinePlot';
import Histogram from './plots/Histogram';
import ViolinPlot from './plots/ViolinPlot';
import Heatmap from './plots/Heatmap';
import CurveChart from './plots/CurveChart';
import ScatterPlot from './plots/ScatterPlot';
import BarChart from './plots/BarChart';

/**
 * Universal Visualization Renderer
 * Routes plot configs to appropriate components based on type
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
