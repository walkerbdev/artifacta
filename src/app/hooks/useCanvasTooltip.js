import { useState, useCallback, useRef, useEffect } from 'react';

/**
 * Hook for managing tooltips on canvas-based plots
 *
 * Usage:
 *   const tooltip = useCanvasTooltip({
 *     canvasRef,
 *     getTooltipData: (canvasX, canvasY, searchRadius) => {
 *       // Your plot-specific logic to find nearest point
 *       // Return { type: 'series', content: {...} } or null
 *     }
 *   });
 *
 *   return (
 *     <>
 *       <canvas ref={canvasRef} />
 *       <PlotTooltip {...tooltip} />
 *     </>
 *   );
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
