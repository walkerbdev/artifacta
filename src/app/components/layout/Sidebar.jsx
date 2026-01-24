import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './Sidebar.scss';

/**
 * Sidebar component for navigation and run selection
 *
 * Collapsible and resizable sidebar containing run browser, filters, and navigation.
 * Primary interface for selecting which experiment runs to analyze.
 *
 * Features:
 * - Collapsible (hide/show with toggle button)
 * - Resizable width (drag right edge)
 * - Backend status indicator
 * - Smooth collapse animation (Framer Motion)
 * - Persistent width across sessions
 *
 * Contains:
 * - Run selector (tree or list view)
 * - Run filters
 * - Project selector
 * - Backend connection status
 *
 * @param {object} props - Component props
 * @param {React.ReactNode} props.children - Sidebar content (RunSelector, filters, etc.)
 * @param {object} props.backendStatus - Backend connection status:
 *   - connected: boolean
 *   - message: string (optional)
 * @param {boolean} props.isCollapsed - Whether sidebar is currently collapsed
 * @param {function} props.onToggle - Callback to toggle collapse state
 * @returns {React.ReactElement} Collapsible, resizable sidebar
 */
export const Sidebar = ({ children, backendStatus, isCollapsed, onToggle }) => {
  const [sidebarWidth, setSidebarWidth] = useState(267);
  const [isResizing, setIsResizing] = useState(false);
  const sidebarRef = useRef(null);

  useEffect(() => {
    if (!isResizing) return;

    /**
     * Handles mouse movement during sidebar resize
     * @param {Event} e - Mouse event object
     * @returns {void}
     */
    const handleMouseMove = (e) => {
      e.preventDefault();
      const newWidth = e.clientX;
      // No restrictions - resize freely
      setSidebarWidth(Math.max(0, newWidth));
    };

    /**
     * Handles mouse up event to end sidebar resize
     * @returns {void}
     */
    const handleMouseUp = () => {
      setIsResizing(false);
      document.body.classList.remove('resizing-sidebar');
    };

    // Prevent text selection during resize
    document.body.classList.add('resizing-sidebar');

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.classList.remove('resizing-sidebar');
    };
  }, [isResizing]);

  return (
    <>
      <motion.aside
        ref={sidebarRef}
        className={`viz-sidebar ${isCollapsed ? 'collapsed' : ''}`}
        initial={{ width: sidebarWidth }}
        animate={{ width: isCollapsed ? 0 : sidebarWidth }}
        transition={{ duration: 0.3 }}
        style={{ width: isCollapsed ? 0 : sidebarWidth }}
      >
        {/* Resize handle */}
        {!isCollapsed && (
          <div
            className="sidebar-resize-handle"
            onMouseDown={(e) => {
              e.preventDefault();
              setIsResizing(true);
            }}
          />
        )}

        {/* Toggle button - positioned at center left edge */}
        {!isCollapsed && (
          <button
            className="sidebar-toggle-btn"
            onClick={onToggle}
            title="Collapse sidebar"
          >
            ‹
          </button>
        )}

      {/* Service Status Indicator */}
      {!isCollapsed && backendStatus && (
        <div className="sidebar-status">
          <div className={`status-indicator ${backendStatus.connected ? 'connected' : 'disconnected'}`}>
            <span className="status-dot"></span>
            <span className="status-text">
              {backendStatus.connected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>
      )}

      {/* Sidebar Content */}
      <AnimatePresence>
        {!isCollapsed && (
          <motion.div
            className="sidebar-content"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            {children}
          </motion.div>
        )}
      </AnimatePresence>
      </motion.aside>

      {/* Collapsed expand button with hover zone */}
      {isCollapsed && (
        <div className="sidebar-expand-zone">
          <button
            className="sidebar-expand-btn"
            onClick={onToggle}
            title="Expand sidebar"
          >
            ›
          </button>
        </div>
      )}
    </>
  );
};
