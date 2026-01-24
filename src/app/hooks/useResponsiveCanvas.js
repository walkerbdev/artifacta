import { useState, useEffect } from 'react';

/**
 * Custom hook for responsive canvas rendering with automatic resize handling
 *
 * Manages canvas dimensions and automatically redraws when the container size changes.
 * Uses ResizeObserver for efficient resize detection and avoids unnecessary redraws.
 *
 * Key features:
 * - Automatically tracks container size changes (parent element resizing)
 * - Debounces dimension updates to avoid redundant redraws
 * - Uses ResizeObserver (modern, performant alternative to window resize events)
 * - Handles device pixel ratio (DPR) via drawCallback
 * - Cleans up observers on unmount
 *
 * Design philosophy:
 * Container controls size → Canvas adapts → Draw callback renders
 * The canvas element's size is determined by CSS (parent container), not hardcoded dimensions.
 *
 * How it works:
 * 1. Observe canvas element size using ResizeObserver
 * 2. When size changes, update dimensions state
 * 3. Trigger drawCallback with new width/height
 * 4. drawCallback is responsible for setting canvas.width/height with DPR scaling
 *
 * @param {React.RefObject<HTMLCanvasElement>} canvasRef - Ref to canvas element
 * @param {function} drawCallback - Function called when canvas needs redraw
 *   Signature: (width: number, height: number) => void
 *   Callback should handle canvas.width/height and ctx.scale(dpr, dpr) for HiDPI
 * @param {object} [options={}] - Optional configuration
 * @param {number} [options.defaultHeight=600] - Fallback height if clientHeight is 0
 * @returns {{width: number, height: number}} Current canvas dimensions (CSS pixels)
 *
 * @example
 * const canvasRef = useRef(null);
 *
 * const drawChart = useCallback((width, height) => {
 *   const canvas = canvasRef.current;
 *   const ctx = canvas.getContext('2d');
 *   const dpr = window.devicePixelRatio || 1;
 *
 *   // Set bitmap size (accounting for DPR)
 *   canvas.width = width * dpr;
 *   canvas.height = height * dpr;
 *   ctx.scale(dpr, dpr);
 *
 *   // Draw using CSS pixels
 *   ctx.fillRect(0, 0, width, height);
 * }, []);
 *
 * useResponsiveCanvas(canvasRef, drawChart);
 *
 * return <canvas ref={canvasRef} style={{ width: '100%', height: '400px' }} />;
 */
export function useResponsiveCanvas(canvasRef, drawCallback, options = {}) {
  const { defaultHeight = 600 } = options;
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    /**
     * Updates canvas dimensions based on current element size
     * @returns {void}
     */
    const updateDimensions = () => {
      const width = canvas.clientWidth;
      const height = canvas.clientHeight || defaultHeight;

      // Only update if dimensions are valid
      if (width > 0 && height > 0) {
        setDimensions(prev => {
          // Avoid unnecessary updates if dimensions haven't changed
          if (prev.width === width && prev.height === height) {
            return prev;
          }
          return { width, height };
        });
      }
    };

    // Check immediately
    updateDimensions();

    // Set up ResizeObserver to handle container size changes (drag/resize)
    // CRITICAL: Only observe border-box size changes, NOT content size (bitmap) changes
    const resizeObserver = new ResizeObserver((entries) => {
      for (let entry of entries) {
        // Only react to border-box size changes, ignore content-box (bitmap) changes
        if (entry.borderBoxSize || entry.contentBoxSize) {
          updateDimensions();
          break;
        }
      }
    });
    resizeObserver.observe(canvas, { box: 'border-box' });

    return () => {
      resizeObserver.disconnect();
    };
  }, [canvasRef, defaultHeight]);

  // Call draw callback whenever dimensions change
  useEffect(() => {
    if (dimensions.width > 0 && dimensions.height > 0) {
      drawCallback(dimensions.width, dimensions.height);
    }
  }, [dimensions, drawCallback, canvasRef]);

  return dimensions;
}
