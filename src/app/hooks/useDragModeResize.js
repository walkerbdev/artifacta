import { useEffect } from 'react';

/**
 * Hook to automatically redraw canvas/component when resized in drag-enabled mode
 * Sets up ResizeObserver only when parent has drag-enabled class to avoid feedback loops
 *
 * @param {React.RefObject} elementRef - Ref to the element to observe (usually canvas)
 * @param {Function} redrawCallback - Function to call when resize is detected
 */
export function useDragModeResize(elementRef, redrawCallback) {
  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;

    const parent = element.closest('.viz-draggable-viz');
    if (!parent) return;

    let resizeObserver = null;

    // Set up or tear down ResizeObserver based on drag-enabled state
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
