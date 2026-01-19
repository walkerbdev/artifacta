import { useState, useCallback, useRef } from 'react';

/**
 * Default layout configuration - defines which components go in which row
 * Each array represents a row, with component keys that should be grouped together
 */
const DEFAULT_LAYOUT = [
  ['trainingData', 'lossCurve2D', 'lossSurface3D'],  // Row 1: Training data + Loss curve/surface
  ['lossHistory', 'accuracyHistory', 'r2History'],   // Row 2: Loss/Accuracy/R² history
  ['f1History', 'precisionRecallHistory'],           // Row 3: F1 and Precision/Recall history
  ['confusionMatrix', 'rocCurve', 'prCurve'],        // Row 4: Classification metrics
  ['actualVsPredicted', 'residualDistribution']      // Row 5: Regression metrics
];

/**
 * Transform-based layout manager - components stay in flexbox flow, use transforms for dragging
 * No absolute positioning mode - everything stays in document flow
 * Dragging applies CSS transforms, drag end reorders DOM if needed
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
