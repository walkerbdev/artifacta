import { useEffect } from 'react';

/**
 * Custom hook for conditional resize observation in drag-enabled mode
 *
 * Automatically redraws canvas/chart elements when resized, but ONLY when drag mode
 * is active. This prevents resize feedback loops and unnecessary redraws during normal
 * layout.
 *
 * Why conditional observation:
 * - Avoids infinite loops (resize → redraw → resize → ...)
 * - Improves performance (no unnecessary redraws in normal mode)
 * - Enables responsive resize handles in drag mode
 *
 * How it works:
 * 1. Uses MutationObserver to watch parent for 'drag-enabled' class changes
 * 2. When drag-enabled added: Attach ResizeObserver to element
 * 3. When drag-enabled removed: Detach ResizeObserver
 * 4. ResizeObserver triggers redrawCallback on size changes
 *
 * @param {React.RefObject} elementRef - Ref to element to observe (typically canvas)
 * @param {function} redrawCallback - Function to call when element resizes
 *   Should redraw the visualization using new dimensions
 * @returns {void}
 *
 * @example
 * const canvasRef = useRef(null);
 * const drawChart = useCallback(() => {
 *   // Redraw chart logic
 * }, []);
 *
 * useDragModeResize(canvasRef, drawChart);
 */
export function useDragModeResize(elementRef, redrawCallback) {
  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;

    const parent = element.closest('.viz-draggable-viz');
    if (!parent) return;

    let resizeObserver = null;

    // Set up or tear down ResizeObserver based on drag-enabled state
    /**
     * Sets up or tears down the ResizeObserver based on drag-enabled state
     * @returns {void}
     */
    const setupResizeObserver = () => {
      const isDragEnabled = parent.classList.contains('drag-enabled');

      if (isDragEnabled && !resizeObserver) {
        resizeObserver = new ResizeObserver(() => {
          redrawCallback();
        });
        resizeObserver.observe(element);
      } else if (!isDragEnabled && resizeObserver) {
        resizeObserver.disconnect();
        resizeObserver = null;
      }
    };

    // Watch for class changes on parent to detect when drag mode is enabled/disabled
    const mutationObserver = new MutationObserver(() => {
      setupResizeObserver();
    });

    mutationObserver.observe(parent, {
      attributes: true,
      attributeFilter: ['class']
    });

    // Initial setup
    setupResizeObserver();

    return () => {
      mutationObserver.disconnect();
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
    };
  }, [elementRef, redrawCallback]);
}
