import React, { useRef } from 'react';
import { motion } from 'framer-motion';
import './DraggableVisualization.scss';

/**
 * Draggable visualization wrapper using transform-based drag system
 *
 * Wraps plot components to make them draggable and resizable while staying in flexbox flow.
 * Uses CSS transforms for smooth 60fps dragging without layout recalculation.
 *
 * Architecture:
 * - Components remain in normal document flow (NOT position: absolute)
 * - During drag: Apply CSS transform for visual feedback (GPU-accelerated)
 * - After drag: Clear transform, optionally reorder DOM
 * - Resize: Uses Framer Motion drag handles
 *
 * Features:
 * - Smooth dragging with transform-based animation
 * - Resize handles (corners and edges)
 * - Close button for removing visualizations
 * - Header with title
 * - Automatic z-index management (dragged element on top)
 * - Integration with layout manager
 *
 * Why transforms instead of absolute positioning:
 * - No layout thrashing (no offsetTop/offsetLeft calculations)
 * - GPU-accelerated (compositing layer)
 * - Maintains flexbox benefits (responsive, auto-sizing)
 * - Easier to revert (just remove transform)
 *
 * @param {object} props - Component props
 * @param {string} props.visualizationKey - Unique identifier for this visualization
 * @param {string} props.title - Display title in header
 * @param {function} props.onClose - Callback when close button clicked
 * @param {function} props.onDragStart - Callback when drag begins (key)
 * @param {function} props.onDrag - Callback during drag (key, deltaX, deltaY)
 * @param {function} props.onDragEnd - Callback when drag ends (key)
 * @param {function} props.onResize - Callback when resized (key, width, height)
 * @param {object} [props.dragTransform] - CSS transform to apply: { x, y }
 * @param {boolean} [props.isDragging] - Whether currently being dragged
 * @param {function} props.registerElement - Register with layout manager (key, element, type)
 * @param {string} props.chartType - Type of chart for registration
 * @param {React.ReactNode} props.children - Plot component to wrap
 * @returns {React.ReactElement} Draggable visualization wrapper
 */
export function DraggableVisualization({
  visualizationKey,
  title,
  onClose,
  onDragStart,
  onDrag,
  onDragEnd,
  onResize,
  dragTransform, // {x, y} transform to apply during drag
  isDragging,    // Whether this element is currently being dragged
  registerElement,
  chartType,
  children
}) {
  const elementRef = useRef(null);
  const [isResizing, setIsResizing] = React.useState(false);
  const [cursorStyle, setCursorStyle] = React.useState('grab');
  const [customSize, setCustomSize] = React.useState(null); // {width, height} when manually resized
  const resizeStartRef = useRef(null);

  const EDGE_THRESHOLD = 8; // pixels from edge to trigger resize

  // Register this element
  React.useEffect(() => {
    if (registerElement && elementRef.current) {
      registerElement(visualizationKey, elementRef.current, chartType);
    }
  }, [registerElement, visualizationKey, chartType]);

  /**
   * Handles the start of a drag operation
   * @returns {void}
   */
  const handleDragStart = () => {
    if (onDragStart) {
      onDragStart(visualizationKey);
    }
  };

  /**
   * Handles drag movement
   * @param {object} _event - The drag event (unused)
   * @param {object} info - Information about the drag state
   * @returns {void}
   */
  const handleDrag = (_event, info) => {
    if (onDrag) {
      // Pass delta from start of drag
      onDrag(visualizationKey, info.offset.x, info.offset.y);
    }
  };

  /**
   * Handles the end of a drag operation
   * @returns {void}
   */
  const handleDragEnd = () => {
    if (onDragEnd) {
      onDragEnd(visualizationKey);
    }
  };

  // Edge detection for resize
  /**
   * Detects which edge of the element the mouse is near
   * @param {object} e - The mouse event
   * @returns {string|null} The edge identifier (n, s, e, w, ne, nw, se, sw) or null
   */
  const getEdgeFromMouse = (e) => {
    if (!elementRef.current) return null;

    const rect = elementRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const nearTop = y < EDGE_THRESHOLD;
    const nearBottom = y > rect.height - EDGE_THRESHOLD;
    const nearLeft = x < EDGE_THRESHOLD;
    const nearRight = x > rect.width - EDGE_THRESHOLD;

    if (nearTop && nearLeft) return 'nw';
    if (nearTop && nearRight) return 'ne';
    if (nearBottom && nearLeft) return 'sw';
    if (nearBottom && nearRight) return 'se';
    if (nearTop) return 'n';
    if (nearBottom) return 's';
    if (nearLeft) return 'w';
    if (nearRight) return 'e';
    return null;
  };

  /**
   * Gets the appropriate cursor style for a given edge
   * @param {string} edge - The edge identifier
   * @returns {string} The CSS cursor value
   */
  const getCursorForEdge = (edge) => {
    const cursors = {
      n: 'ns-resize',
      s: 'ns-resize',
      e: 'ew-resize',
      w: 'ew-resize',
      ne: 'nesw-resize',
      sw: 'nesw-resize',
      nw: 'nwse-resize',
      se: 'nwse-resize'
    };
    return cursors[edge] || 'default';
  };

  /**
   * Handles mouse movement to update cursor style based on edge proximity
   * @param {object} e - The mouse event
   * @returns {void}
   */
  const handleMouseMove = (e) => {
    if (isResizing || isDragging) return;

    const edge = getEdgeFromMouse(e);
    setCursorStyle(edge ? getCursorForEdge(edge) : 'grab');
  };

  /**
   * Handles mouse down to initiate resizing
   * @param {object} e - The mouse event
   * @returns {void}
   */
  const handleMouseDown = (e) => {
    // Don't handle if clicking on close button or interactive elements
    if (e.target.closest('.viz-viz-close-btn, select, button, input, textarea, a')) {
      return;
    }

    const edge = getEdgeFromMouse(e);
    if (!edge) return; // Not on edge, let drag handle it

    e.stopPropagation();
    e.preventDefault();

    // Start resizing - capture current size (including auto-calculated height)
    const rect = elementRef.current.getBoundingClientRect();

    resizeStartRef.current = {
      width: rect.width,
      height: rect.height, // Captures the actual rendered height, even if CSS is 'auto'
      mouseX: e.clientX,
      mouseY: e.clientY,
      edge
    };

    // If we don't have a custom size yet, set it to current size to enable vertical resizing
    if (!customSize) {
      setCustomSize({ width: rect.width, height: rect.height });
    }

    setIsResizing(true);
  };

  // Handle resize move
  React.useEffect(() => {
    if (!isResizing) return;

    /**
     * Handles mouse movement during resize operation
     * @param {object} e - The mouse event
     * @returns {void}
     */
    const handleResizeMove = (e) => {
      if (!resizeStartRef.current) return;

      const deltaX = e.clientX - resizeStartRef.current.mouseX;
      const deltaY = e.clientY - resizeStartRef.current.mouseY;
      const edge = resizeStartRef.current.edge;

      let newWidth = resizeStartRef.current.width;
      let newHeight = resizeStartRef.current.height;

      // Adjust based on which edge is being dragged
      if (edge.includes('e')) {
        newWidth = resizeStartRef.current.width + deltaX;
      }
      if (edge.includes('w')) {
        newWidth = resizeStartRef.current.width - deltaX;
      }
      if (edge.includes('s')) {
        newHeight = resizeStartRef.current.height + deltaY;
      }
      if (edge.includes('n')) {
        newHeight = resizeStartRef.current.height - deltaY;
      }

      // Apply minimum sizes
      newWidth = Math.max(300, newWidth);
      newHeight = Math.max(200, newHeight);

      setCustomSize({ width: newWidth, height: newHeight });

      if (onResize) {
        onResize(visualizationKey, newWidth, newHeight);
      }
    };

    /**
     * Handles the end of a resize operation
     * @returns {void}
     */
    const handleResizeEnd = () => {
      setIsResizing(false);
      resizeStartRef.current = null;
    };

    document.addEventListener('mousemove', handleResizeMove);
    document.addEventListener('mouseup', handleResizeEnd);

    return () => {
      document.removeEventListener('mousemove', handleResizeMove);
      document.removeEventListener('mouseup', handleResizeEnd);
    };
  }, [isResizing, onResize, visualizationKey]);

  // Calculate style - apply transform if dragging, custom size if resized
  const divStyle = {
    cursor: isResizing ? getCursorForEdge(resizeStartRef.current?.edge) : (isDragging ? 'grabbing' : cursorStyle),
    // Use custom size if manually resized, otherwise use defaults
    width: customSize ? `${customSize.width}px` :
           (chartType === 'line' || chartType === 'scatter' || chartType === 'parallel_coordinates' ||
            chartType === 'parameter_importance' || chartType === 'comparison_scatter' ||
            chartType === 'parameter_distribution' || chartType === 'heatmap' ? '644px' : '500px'),
    height: customSize ? `${customSize.height}px` : 'fit-content',
    minHeight: customSize ? 'unset' : '200px',
    boxSizing: 'border-box',
    // Apply transform if dragging
    ...(dragTransform && {
      transform: `translate(${dragTransform.x}px, ${dragTransform.y}px)`,
      zIndex: 1000, // Bring to front when dragging
      transition: 'none' // No transition during drag
    })
  };

  return (
    <motion.div
      ref={elementRef}
      className="viz-draggable-viz"
      drag={!isResizing} // Disable drag when resizing
      dragMomentum={false}
      dragElastic={0}
      dragTransition={{ power: 0, timeConstant: 0 }}
      onDragStart={handleDragStart}
      onDrag={handleDrag}
      onDragEnd={handleDragEnd}
      onMouseMove={handleMouseMove}
      onMouseDown={handleMouseDown}
      style={divStyle}
    >
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'visible'
        }}
      >
        {title && <h4 className="viz-viz-title">{title}</h4>}
        <div style={{
          flex: 1,
          minHeight: 0,
          width: '100%'
        }}>
          {children}
        </div>
      </div>

      {/* Close button */}
      {onClose && (
        <button
          className="viz-viz-close-btn"
          onClick={() => onClose(visualizationKey)}
          aria-label="Close visualization"
        >
          ×
        </button>
      )}
    </motion.div>
  );
}
