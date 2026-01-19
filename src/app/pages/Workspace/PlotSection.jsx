import React from 'react';
import { HiChevronDown, HiChevronUp } from 'react-icons/hi';
import { useLayoutManager } from '@/app/hooks';
import { DraggableVisualization } from '@/app/components/visualizations/DraggableVisualization';
import UniversalVisualizationRenderer from '@/app/components/visualizations/UniversalVisualizationRenderer';

/**
 * PlotSection - Collapsible section with draggable plots
 * Each section manages dragging/positioning for its own plots
 *
 * React key includes dataset count to prevent canvas reuse in multi-run mode
 *
 * Problem: When multiple runs are selected, line plot IDs remain constant (e.g., "Loss_line")
 * while the data changes (1 dataset -> 2 datasets -> 3 datasets). Without dataset count
 * in the React key, React's reconciliation reuses the same component instance, which means
 * the canvas element and its ref are also reused. This caused multiple plots to draw on the
 * same physical canvas, resulting in overlapping axis labels and incorrect zoom behavior.
 *
 * Solution: Include dataset count in the React key (e.g., "Loss_line_4_2" for 2 datasets).
 * When the number of datasets changes, React creates a fresh component with a new canvas.
 */
export const PlotSection = ({
  sectionName,
  plots
}) => {
  // Collapsible state for this section
  const [isCollapsed, setIsCollapsed] = React.useState(false);

  // Visibility state for individual plots
  const [hiddenPlots, setHiddenPlots] = React.useState(new Set());

  // Each section has its OWN layout manager with its OWN boundaries
  const {
    dragTransforms,
    draggingKey,
    handleDragStart,
    handleDrag,
    handleDragEnd,
    handleResize,
    registerElement,
    registerContainer,
    customLabels,
    updateLabels
  } = useLayoutManager();

  const togglePlotVisibility = (plotId) => {
    setHiddenPlots(prev => {
      const next = new Set(prev);
      if (next.has(plotId)) {
        next.delete(plotId);
      } else {
        next.add(plotId);
      }
      return next;
    });
  };

  return (
    <div style={{ width: '100%', marginBottom: '48px', position: 'relative' }}>
      {/* Section header */}
      <h3 style={{
        fontSize: '18px',
        fontWeight: '600',
        color: '#333',
        margin: '0 0 24px 0',
        borderBottom: '2px solid #e0e0e0',
        paddingBottom: '8px',
        paddingRight: '40px' // Make room for button
      }}>
        {sectionName}
      </h3>

      {/* Collapse button */}
      <button
        style={{
          position: 'absolute',
          top: '-6px',
          right: '8px',
          width: '28px',
          height: '28px',
          borderRadius: '50%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'rgba(255, 255, 255, 0.95)',
          border: '1px solid #e5e7eb',
          color: '#9ca3af',
          cursor: 'pointer',
          fontSize: '18px',
          transition: 'all 0.2s ease',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
          zIndex: 10
        }}
        onClick={() => setIsCollapsed(!isCollapsed)}
        onMouseEnter={(e) => {
          e.currentTarget.style.background = '#f3f4f6';
          e.currentTarget.style.color = '#667eea';
          e.currentTarget.style.borderColor = '#d1d5db';
          e.currentTarget.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.15)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.background = 'rgba(255, 255, 255, 0.95)';
          e.currentTarget.style.color = '#9ca3af';
          e.currentTarget.style.borderColor = '#e5e7eb';
          e.currentTarget.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.1)';
        }}
        onMouseDown={(e) => {
          e.currentTarget.style.transform = 'scale(0.95)';
        }}
        onMouseUp={(e) => {
          e.currentTarget.style.transform = 'scale(1)';
        }}
        title={isCollapsed ? "Expand section" : "Collapse section"}
      >
        {isCollapsed ? <HiChevronDown /> : <HiChevronUp />}
      </button>

      {/* Plots container with its OWN layout manager boundaries */}
      {!isCollapsed && (
        <div
          ref={registerContainer}
          className="viz-viz-row"
          style={{
            position: 'relative',
            minHeight: '100px'
          }}
        >
          {plots.map((plotConfig, index) => {
            if (hiddenPlots.has(plotConfig.id)) return null;

            // CRITICAL: Include dataset count in React key to prevent canvas reuse
            // When multiple runs are selected, plot IDs stay the same (e.g., "Loss_line")
            // but the number of datasets changes (1 dataset -> 2 datasets -> 3 datasets).
            // Without proper key changes, React reuses the same component instance.
            const datasetCount = plotConfig.data?.datasets?.length || 0;
            const uniqueKey = `${plotConfig.id}_${index}_${datasetCount}`;

            return (
              <DraggableVisualization
                key={uniqueKey}
                visualizationKey={plotConfig.id}
                title={plotConfig.title}
                onClose={togglePlotVisibility}
                registerElement={registerElement}
                onDragStart={handleDragStart}
                onDrag={handleDrag}
                onDragEnd={handleDragEnd}
                onResize={handleResize}
                dragTransform={dragTransforms[plotConfig.id]}
                isDragging={draggingKey === plotConfig.id}
                customLabels={customLabels[plotConfig.id]}
                onUpdateLabels={(labels) => updateLabels(plotConfig.id, labels)}
                chartType={plotConfig.type}
              >
                <UniversalVisualizationRenderer plotConfig={plotConfig} />
              </DraggableVisualization>
            );
          })}
        </div>
      )}
    </div>
  );
};
