import { useState, useEffect } from 'react';

/**
 * Generic hook for responsive canvas rendering
 * Handles canvas sizing, ResizeObserver, and automatic redraw on dimension changes
 *
 * SIMPLE RULE: Container controls size, canvas adapts. No height constraints.
 *
 * @param {React.RefObject} canvasRef - Reference to canvas element
 * @param {Function} drawCallback - Function to call when canvas needs redraw
 *   - Parameters: (width, height) => void
 * @param {Object} options - Optional configuration
 * @param {number} options.defaultHeight - Default canvas height before layout calculation (default: 600)
 * @returns {{ width: number, height: number }} - Current canvas dimensions
 */
export function useResponsiveCanvas(canvasRef, drawCallback, options = {}) {
  const { defaultHeight = 600 } = options;
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

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
