import React, { useRef, useCallback } from 'react';
import { getChartDimensions, drawAxes, drawGridLines, getChartFont } from '@/app/hooks/useCanvasSetup';
import { useResponsiveCanvas } from '@/app/hooks/useResponsiveCanvas';
import { useCanvasTooltip } from '@/app/hooks/useCanvasTooltip';
import PlotTooltip from '../shared/PlotTooltip';
import { toTitleCase } from '@/core/utils/formatters';
import { getChartColor, CHART_PADDING } from '@/core/utils/constants';

/**
 * Curve Chart component for performance curves (ROC, PR, calibration)
 *
 * Generic field-agnostic curve visualization that auto-detects axes from data.
 * Commonly used for ROC curves (with AUC), precision-recall curves, and calibration plots.
 *
 * Features:
 * - Auto-detects X/Y field names from data structure
 * - Multi-curve overlay with different colors
 * - Optional diagonal reference line (ROC curves)
 * - Metric display (e.g., AUC = 0.95)
 * - Interactive tooltips
 * - Custom axis labels
 * - HiDPI display support
 *
 * Data format (single curve):
 * ```
 * {
 *   data: [
 *     { fpr: 0.0, tpr: 0.0 },
 *     { fpr: 0.1, tpr: 0.7 },
 *     { fpr: 1.0, tpr: 1.0 }
 *   ],
 *   metric: 0.95,
 *   metricLabel: "AUC"
 * }
 * ```
 *
 * Data format (multi-curve):
 * ```
 * {
 *   curves: [
 *     { label: "Model A", points: [...], metric: { name: "AUC", value: 0.95 } },
 *     { label: "Model B", points: [...], metric: { name: "AUC", value: 0.92 } }
 *   ]
 * }
 * ```
 *
 * @param {object} props - Component props
 * @param {Array<object>|object} props.data - Curve data (array of points or object with curves)
 * @param {Array<object>} [props.curves] - Multi-curve data with labels
 * @param {string} [props.xField] - X-axis field (auto-detected from first point)
 * @param {string} [props.yField] - Y-axis field (auto-detected from first point)
 * @param {string} [props.xLabel] - X-axis label (auto-generated if not provided)
 * @param {string} [props.yLabel] - Y-axis label (auto-generated if not provided)
 * @param {number} [props.metric] - Metric value to display (e.g., 0.95 for AUC)
 * @param {string} [props.metricLabel] - Metric name (e.g., "AUC", "Average Precision")
 * @param {boolean} [props.showDiagonal=false] - Show y=x diagonal line (for ROC)
 * @returns {React.ReactElement|null} Canvas-based curve chart with tooltip
 */
const CurveChart = ({
  data,           // Array of objects with numeric fields: [{x, y}, ...] OR {curves: [...]} for multi-run
  curves,         // Multi-run: Array of curve objects with labels [{label, points, metric}, ...]
  xField,         // X-axis field name (auto-detected if not provided)
  yField,         // Y-axis field name (auto-detected if not provided)
  xLabel,         // X-axis label (auto-generated from xField if not provided)
  yLabel,         // Y-axis label (auto-generated from yField if not provided)
  metric,         // Optional metric value to display (e.g., AUC)
  metricLabel,    // Label for metric (e.g., "AUC")
  showDiagonal,   // Show diagonal reference line (for ROC curves)
  rocData,
  prData,
  visibleROC,
  visiblePR,
  customXLabelROC,
  customYLabelROC,
  customXLabelPR,
  customYLabelPR
}) => {
  const canvasRef = useRef(null);
  const plotDataRef = useRef(null);

  // Multi-run support: check if data has curves array
  const isMultiRun = curves && Array.isArray(curves) && curves.length > 0;

  const curveData = isMultiRun ? null : (data || (rocData?.points) || (prData?.points));
  const legacyMetric = rocData?.auc || prData?.avgPrecision;
  const legacyXLabel = customXLabelROC || customXLabelPR;
  const legacyYLabel = customYLabelROC || customYLabelPR;
  const isROC = !!rocData || showDiagonal;

  /**
   * Auto-detect field names from data
   * @returns {object} Object with x and y field names
   */
  const detectFields = useCallback(() => {
    if (!curveData || curveData.length === 0) return { x: null, y: null };

    const sample = curveData[0];
    const numericFields = Object.keys(sample).filter(k => typeof sample[k] === 'number');

    // Default: use first two numeric fields
    return {
      x: xField || numericFields[0] || 'x',
      y: yField || numericFields[1] || 'y'
    };
  }, [curveData, xField, yField]);

  /**
   * Draws the curve chart on the canvas
   * @param {number} width - Canvas width
   * @param {number} height - Canvas height
   */
  const drawCurve = useCallback((width, height) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (!isMultiRun && (!curveData || curveData.length === 0)) return;
    if (isMultiRun && (!curves || curves.length === 0)) return;

    // Setup canvas with DPR
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);

    // Clear canvas
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    // Canvas is JUST the plot area - legend is HTML outside
    const padding = CHART_PADDING;

    // Use ALL available space - no height constraints
    const { chartWidth, chartHeight } = getChartDimensions(width, height, padding);

    if (isMultiRun) {
      // Multi-run mode
      const finalXLabel = xLabel || 'X';
      const finalYLabel = yLabel || 'Y';

      // Find data ranges across all curves
      const allPoints = curves.flatMap(c => c.points);
      const xValues = allPoints.map(p => p.x);
      const yValues = allPoints.map(p => p.y);
      const xMin = Math.min(...xValues);
      const xMax = Math.max(...xValues);
      const yMin = Math.min(...yValues);
      const yMax = Math.max(...yValues);

      /**
       * Converts data X value to canvas X coordinate
       * @param {number} val - Data X value
       * @returns {number} Canvas X coordinate
       */
      const toCanvasX = (val) => padding.left + ((val - xMin) / (xMax - xMin || 1)) * chartWidth;
      /**
       * Converts data Y value to canvas Y coordinate
       * @param {number} val - Data Y value
       * @returns {number} Canvas Y coordinate
       */
      const toCanvasY = (val) => padding.top + chartHeight - ((val - yMin) / (yMax - yMin || 1)) * chartHeight;

      // Draw axes and grid
      drawAxes(ctx, width, height, padding, finalXLabel, finalYLabel);
      drawGridLines(ctx, width, height, padding, 5);

      // Diagonal reference line
      if (showDiagonal && xMin === 0 && xMax === 1 && yMin === 0 && yMax === 1) {
        ctx.strokeStyle = '#999';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(toCanvasX(0), toCanvasY(0));
        ctx.lineTo(toCanvasX(1), toCanvasY(1));
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Draw each curve
      curves.forEach((curve, idx) => {
        const color = getChartColor(idx);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.beginPath();

        curve.points.forEach((point, i) => {
          const x = toCanvasX(point.x);
          const y = toCanvasY(point.y);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      });

      // Store data for tooltip (multi-run) - on-the-fly transform
      plotDataRef.current = {
        isMultiRun: true,
        curves,
        xMin,
        xMax,
        yMin,
        yMax,
        padding,
        chartWidth,
        chartHeight,
        xLabel: finalXLabel,
        yLabel: finalYLabel,
        metricLabel
      };
    } else {
      // Single-run mode (original code)
      const fields = detectFields();
      if (!fields.x || !fields.y) return;

      const finalXLabel = xLabel || legacyXLabel || toTitleCase(fields.x);
      const finalYLabel = yLabel || legacyYLabel || toTitleCase(fields.y);

      const xValues = curveData.map(p => p[fields.x]);
      const yValues = curveData.map(p => p[fields.y]);
      const xMin = Math.min(...xValues);
      const xMax = Math.max(...xValues);
      const yMin = Math.min(...yValues);
      const yMax = Math.max(...yValues);

      /**
       * Converts data X value to canvas X coordinate
       * @param {number} val - Data X value
       * @returns {number} Canvas X coordinate
       */
      const toCanvasX = (val) => padding.left + ((val - xMin) / (xMax - xMin || 1)) * chartWidth;
      /**
       * Converts data Y value to canvas Y coordinate
       * @param {number} val - Data Y value
       * @returns {number} Canvas Y coordinate
       */
      const toCanvasY = (val) => padding.top + chartHeight - ((val - yMin) / (yMax - yMin || 1)) * chartHeight;

      drawAxes(ctx, width, height, padding, finalXLabel, finalYLabel);
      drawGridLines(ctx, width, height, padding, 5);

      if (isROC && xMin === 0 && xMax === 1 && yMin === 0 && yMax === 1) {
        ctx.strokeStyle = '#999';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(toCanvasX(0), toCanvasY(0));
        ctx.lineTo(toCanvasX(1), toCanvasY(1));
        ctx.stroke();
        ctx.setLineDash([]);
      }

      ctx.strokeStyle = '#2563eb';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      curveData.forEach((point, i) => {
        const x = toCanvasX(point[fields.x]);
        const y = toCanvasY(point[fields.y]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      const finalMetric = metric ?? legacyMetric;
      const finalMetricLabel = metricLabel || (isROC ? 'AUC' : 'Avg Precision');

      if (finalMetric !== undefined && finalMetric !== null) {
        ctx.fillStyle = '#2563eb';
        ctx.font = getChartFont('metric');
        ctx.fillText(`${finalMetricLabel} = ${finalMetric.toFixed(3)}`, width / 2, 35);
      }

      // Store data for tooltip (single-run) - on-the-fly transform
      plotDataRef.current = {
        isMultiRun: false,
        curveData,
        fields,
        xMin,
        xMax,
        yMin,
        yMax,
        padding,
        chartWidth,
        chartHeight,
        xLabel: finalXLabel,
        yLabel: finalYLabel,
        metric: finalMetric,
        metricLabel: finalMetricLabel
      };
    }

    // No return needed - container controls size, plot adapts
  }, [curveData, curves, isMultiRun, xLabel, yLabel, metric, metricLabel, isROC, showDiagonal, detectFields, legacyXLabel, legacyYLabel, legacyMetric]);

  // Use responsive canvas hook - handles sizing and redraw automatically
  useResponsiveCanvas(canvasRef, drawCurve);

  /**
   * Tooltip logic: find nearest point on curves (on-the-fly transform)
   * @param {number} mouseX - Mouse X coordinate
   * @param {number} mouseY - Mouse Y coordinate
   * @param {number} searchRadius - Search radius for finding nearest point
   * @returns {object|null} Tooltip data or null if no point found
   */
  const getTooltipData = useCallback((mouseX, mouseY, searchRadius) => {
    if (!plotDataRef.current) return null;

    const pd = plotDataRef.current;
    const { padding, chartWidth, chartHeight, xMin, xMax, yMin, yMax } = pd;

    /**
     * Converts data X value to canvas X coordinate
     * @param {number} val - Data X value
     * @returns {number} Canvas X coordinate
     */
    const toCanvasX = (val) => padding.left + ((val - xMin) / (xMax - xMin || 1)) * chartWidth;
    /**
     * Converts data Y value to canvas Y coordinate
     * @param {number} val - Data Y value
     * @returns {number} Canvas Y coordinate
     */
    const toCanvasY = (val) => padding.top + chartHeight - ((val - yMin) / (yMax - yMin || 1)) * chartHeight;

    let nearestPoint = null;
    let minDistance = searchRadius;
    let nearestLabel = null;

    if (pd.isMultiRun) {
      // Multi-run: check all curves
      pd.curves.forEach(curve => {
        curve.points.forEach(point => {
          const canvasX = toCanvasX(point.x);
          const canvasY = toCanvasY(point.y);
          const distance = Math.sqrt(
            Math.pow(canvasX - mouseX, 2) +
            Math.pow(canvasY - mouseY, 2)
          );

          if (distance < minDistance) {
            minDistance = distance;
            nearestPoint = point;
            nearestLabel = `${curve.label}${curve.metric ? ` (${pd.metricLabel}: ${curve.metric.toFixed(3)})` : ''}`;
          }
        });
      });
    } else {
      // Single-run: check points
      pd.curveData?.forEach(point => {
        const xVal = point[pd.fields.x];
        const yVal = point[pd.fields.y];
        const canvasX = toCanvasX(xVal);
        const canvasY = toCanvasY(yVal);
        const distance = Math.sqrt(
          Math.pow(canvasX - mouseX, 2) +
          Math.pow(canvasY - mouseY, 2)
        );

        if (distance < minDistance) {
          minDistance = distance;
          nearestPoint = { x: xVal, y: yVal };
        }
      });
    }

    if (!nearestPoint) return null;

    return {
      type: 'curve',
      content: {
        x: nearestPoint.x,
        y: nearestPoint.y,
        xLabel: pd.xLabel,
        yLabel: pd.yLabel,
        metric: nearestLabel || (pd.metric ? `${pd.metricLabel}: ${pd.metric.toFixed(3)}` : undefined)
      }
    };
  }, []);

  const tooltip = useCanvasTooltip({
    canvasRef,
    getTooltipData,
    searchRadius: 20
  });

  const shouldShow = (visibleROC !== false && visiblePR !== false) && ((curveData && curveData.length > 0) || isMultiRun);

  if (!shouldShow) {
    return null;
  }

  // Build legend items for HTML rendering (multi-run only)
  const legendItems = isMultiRun && curves ? curves.map((curve, idx) => {
    const metricText = curve.metric !== undefined ? ` (${metricLabel || 'AUC'}: ${curve.metric.toFixed(3)})` : '';
    return {
      label: `${curve.label}${metricText}`,
      color: getChartColor(idx)
    };
  }) : null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', width: '100%', height: '100%' }}>
      {/* HTML Legend - auto-wraps and expands (multi-run only) */}
      {legendItems && legendItems.length > 0 && (
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '12px',
          padding: '10px 10px 5px 10px',
          fontSize: '13px',
          flexShrink: 0
        }}>
          {legendItems.map((item, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <div style={{
                width: '16px',
                height: '3px',
                backgroundColor: item.color,
                borderRadius: '1px'
              }} />
              <span>{item.label}</span>
            </div>
          ))}
        </div>
      )}

      {/* Canvas - simple approach */}
      <div style={{ flex: 1, minHeight: 0, padding: '0 10px 10px 10px' }}>
        <canvas
          ref={canvasRef}
          className="viz-canvas-confusion"
          style={{
            width: '100%',
            height: '100%',
            maxWidth: '100%',
            maxHeight: '100%',
            display: 'block',
            cursor: 'crosshair'
          }}
        />
      </div>

      <PlotTooltip {...tooltip} />
    </div>
  );
};

export default CurveChart;
