import { useState, useCallback, useRef } from 'react';

/**
 * Default layout configuration - defines which visualization components go in which row
 *
 * Each array represents a row of visualizations that should be grouped together.
 * This provides a sensible default organization for ML experiment visualizations.
 */
const DEFAULT_LAYOUT = [
  ['trainingData', 'lossCurve2D', 'lossSurface3D'],  // Row 1: Training data + Loss curve/surface
  ['lossHistory', 'accuracyHistory', 'r2History'],   // Row 2: Loss/Accuracy/R² history
  ['f1History', 'precisionRecallHistory'],           // Row 3: F1 and Precision/Recall history
  ['confusionMatrix', 'rocCurve', 'prCurve'],        // Row 4: Classification metrics
  ['actualVsPredicted', 'residualDistribution']      // Row 5: Regression metrics
];

/**
 * Custom hook for managing draggable/resizable visualization layout
 *
 * Provides a transform-based drag-and-drop system that keeps components in normal
 * document flow (using flexbox) while applying CSS transforms during drag operations.
 * This approach avoids the complexity of absolute positioning while still allowing
 * smooth 60fps drag interactions.
 *
 * Architecture:
 * - Components remain in flexbox flow (not position: absolute)
 * - During drag: Apply CSS transform for visual feedback
 * - After drag: Clear transform, optionally reorder DOM elements
 * - Resize: Track dimensions per-component (future: persist to localStorage)
 * - Custom labels: Store user-edited axis labels/titles per-visualization
 *
 * Key features:
 * - Smooth dragging using CSS transforms (GPU-accelerated)
 * - No layout thrashing (no position/size calculations during drag)
 * - Automatic z-index management (dragged element on top)
 * - Custom label persistence per visualization
 * - Element and container registration system
 *
 * State management:
 * - dragTransforms: CSS translate values for currently dragging elements
 * - draggingKey: Which visualization is being dragged (for styling)
 * - customLabels: User-edited labels/titles per visualization
 *
 * @param {Array<Array<string>>} [layoutConfig=DEFAULT_LAYOUT] - Row-based layout configuration
 *   Each nested array defines visualizations that should be grouped in a row
 * @returns {object} Layout manager interface:
 *   - dragTransforms: object - CSS transform values for dragging elements
 *   - draggingKey: string|null - Key of currently dragging element
 *   - handleDragStart: function - Start drag operation
 *   - handleDrag: function - Update drag position
 *   - handleDragEnd: function - Finish drag, clear transforms
 *   - handleResize: function - Handle resize events
 *   - registerElement: function - Register a visualization element
 *   - registerContainer: function - Register the container element
 *   - updateLabels: function - Update custom labels for a visualization
 *   - customLabels: object - Map of visualization keys to custom label objects
 *   - layoutConfig: array - The layout configuration being used
 *
 * @example
 * const {
 *   dragTransforms,
 *   draggingKey,
 *   handleDragStart,
 *   handleDrag,
 *   handleDragEnd,
 *   registerElement,
 *   registerContainer
 * } = useLayoutManager();
 *
 * return (
 *   <div ref={registerContainer}>
 *     <DraggableVisualization
 *       visualizationKey="loss-plot"
 *       onDragStart={handleDragStart}
 *       onDrag={handleDrag}
 *       onDragEnd={handleDragEnd}
 *       dragTransform={dragTransforms['loss-plot']}
 *       isDragging={draggingKey === 'loss-plot'}
 *       registerElement={registerElement}
 *     >
 *       <LinePlot data={data} />
 *     </DraggableVisualization>
 *   </div>
 * );
 */
export function useLayoutManager(layoutConfig = DEFAULT_LAYOUT) {
  const containerRef = useRef(null);
  const elementRefs = useRef({});

  // Track transforms for drag operations (not positions)
  const [dragTransforms, setDragTransforms] = useState({});

  // Track which element is currently being dragged
  const [draggingKey, setDraggingKey] = useState(null);

  // Track custom labels for all components
  const [customLabels, setCustomLabels] = useState({});

  // Register element refs
  const registerElement = useCallback((key, element, _chartType) => {
    if (element) {
      elementRefs.current[key] = element;
    }
  }, []);

  // Register container ref
  const registerContainer = useCallback((element) => {
    if (element) {
      containerRef.current = element;
      element.style.position = 'relative';
      element.style.minHeight = '200px';
    }
  }, []);

  // Start drag - record which element is being dragged
  const handleDragStart = useCallback((key) => {
    setDraggingKey(key);
  }, []);

  // Handle drag - update transform for visual feedback
  const handleDrag = useCallback((key, deltaX, deltaY) => {
    setDragTransforms(prev => ({
      ...prev,
      [key]: { x: deltaX, y: deltaY }
    }));
  }, []);

  // Handle drag end - clear transform and optionally reorder DOM
  const handleDragEnd = useCallback((key) => {
    // Clear transform
    setDragTransforms(prev => {
      const next = { ...prev };
      delete next[key];
      return next;
    });
    setDraggingKey(null);
  }, []);

  // Handle resize
  const handleResize = useCallback((_key, _newWidth, _newHeight) => {
  }, []);

  // Update custom labels for a specific visualization
  const updateLabels = useCallback((visualizationKey, field, value) => {
    setCustomLabels(prev => ({
      ...prev,
      [visualizationKey]: {
        ...prev[visualizationKey],
        [field]: value
      }
    }));
  }, []);

  return {
    dragTransforms, // CSS transforms for currently dragging elements
    draggingKey,    // Which element is currently being dragged
    handleDragStart,
    handleDrag,
    handleDragEnd,
    handleResize,
    registerElement,
    registerContainer,
    layoutConfig,
    customLabels,
    updateLabels
  };
}
