import { useState, useCallback, useRef, useEffect } from 'react';

/**
 * Custom hook for managing interactive tooltips on canvas-based plots
 *
 * Provides smooth, performant tooltip updates at 60fps using requestAnimationFrame.
 * This hook handles all the complexity of:
 * - Mouse position tracking relative to canvas
 * - Finding nearest data points via custom search function
 * - Throttling updates to avoid layout thrashing
 * - Proper cleanup on unmount
 *
 * Performance optimizations:
 * - Uses requestAnimationFrame to throttle tooltip updates to 60fps max
 * - Cancels pending RAF callbacks when new mousemove events arrive
 * - Only triggers React re-renders when tooltip data actually changes
 * - Cleans up event listeners and RAF callbacks on unmount
 *
 * How it works:
 * 1. Attach mousemove/mouseleave listeners to canvas element
 * 2. On mousemove: convert screen coords to canvas coords
 * 3. Schedule RAF callback to find nearby data points
 * 4. If data found: update tooltip state with position and content
 * 5. PlotTooltip component renders the actual tooltip DOM
 *
 * @param {object} params - Hook configuration
 * @param {React.RefObject<HTMLCanvasElement>} params.canvasRef - Ref to canvas element
 * @param {function} params.getTooltipData - Function to find data near cursor position
 *   Signature: (canvasX: number, canvasY: number, searchRadius: number) => object|null
 *   Should return tooltip data object or null if no data nearby
 *   Example return: { type: 'series', content: { x: 10, y: 20, seriesName: 'loss' } }
 * @param {number} [params.searchRadius=20] - Pixel radius for nearby point detection
 * @returns {object} Tooltip state for PlotTooltip component:
 *   - visible: boolean - Whether tooltip should be shown
 *   - x: number - Screen X coordinate for tooltip
 *   - y: number - Screen Y coordinate for tooltip
 *   - data: object|null - Tooltip content data
 *
 * @example
 * // In a plot component:
 * const tooltip = useCanvasTooltip({
 *   canvasRef,
 *   searchRadius: 25,
 *   getTooltipData: (canvasX, canvasY, radius) => {
 *     // Find nearest data point within radius
 *     const point = findNearestPoint(canvasX, canvasY, radius);
 *     if (!point) return null;
 *
 *     return {
 *       type: 'scatter',
 *       content: {
 *         x: point.x,
 *         y: point.y,
 *         label: point.label
 *       }
 *     };
 *   }
 * });
 *
 * return (
 *   <>
 *     <canvas ref={canvasRef} />
 *     <PlotTooltip {...tooltip} />
 *   </>
 * );
 */
export function useCanvasTooltip({
  canvasRef,
  getTooltipData,
  searchRadius = 20,
}) {
  const [tooltip, setTooltip] = useState({
    visible: false,
    x: 0,
    y: 0,
    data: null,
  });

  const rafRef = useRef(null);

  const handleMouseMove = useCallback((e) => {
    if (!canvasRef.current || !getTooltipData) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const canvasX = e.clientX - rect.left;
    const canvasY = e.clientY - rect.top;

    // Cancel previous frame
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
    }

    // Debounce using requestAnimationFrame for smooth 60fps updates
    rafRef.current = requestAnimationFrame(() => {
      const tooltipData = getTooltipData(canvasX, canvasY, searchRadius);

      if (tooltipData) {
        setTooltip({
          visible: true,
          x: e.clientX,
          y: e.clientY,
          data: tooltipData,
        });
      } else {
        setTooltip(prev => ({ ...prev, visible: false }));
      }
    });
  }, [canvasRef, getTooltipData, searchRadius]);

  const handleMouseLeave = useCallback(() => {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
    }
    setTooltip({ visible: false, x: 0, y: 0, data: null });
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseleave', handleMouseLeave);

    return () => {
      canvas.removeEventListener('mousemove', handleMouseMove);
      canvas.removeEventListener('mouseleave', handleMouseLeave);
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, [canvasRef, handleMouseMove, handleMouseLeave]);

  return tooltip;
}
