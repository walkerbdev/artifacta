import React from 'react';
import { HiChevronDown, HiChevronUp } from 'react-icons/hi';
import { useLayoutManager } from '@/app/hooks';
import { DraggableVisualization } from '@/app/components/visualizations/DraggableVisualization';
import UniversalVisualizationRenderer from '@/app/components/visualizations/UniversalVisualizationRenderer';

/**
 * PlotSection - Collapsible section with draggable plots
 * Each section manages dragging/positioning for its own plots
 * @param {object} props - Component props
 * @param {string} props.sectionName - Name of the section
 * @param {Array<object>} props.plots - Array of plot configurations to render
 * @returns {React.ReactElement} The plot section component
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

  /**
   * Toggles the visibility of a specific plot by ID
   * @param {string} plotId - The ID of the plot to toggle
   * @returns {void}
   */
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

            return (
              <DraggableVisualization
                key={`${plotConfig.id}_${index}`}
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
