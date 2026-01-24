import React from 'react';
import { HiChevronDown, HiChevronUp } from 'react-icons/hi';

/**
 * Collapsible toggle button for expand/collapse interactions
 *
 * Reusable UI button used throughout the app for collapsing sections.
 * Primarily used in RunSelector for collapsing run detail tables.
 *
 * Features:
 * - Animated chevron icon (up/down)
 * - Hover effects (color change, shadow)
 * - Click animation (scale down on press)
 * - Tooltip showing current state
 * - Circular design with subtle shadow
 *
 * @param {object} props - Component props
 * @param {boolean} props.isCollapsed - Current collapsed state
 * @param {function} props.onClick - Click handler to toggle state
 * @param {string} [props.title] - Custom tooltip (defaults to "Expand/Collapse table")
 * @returns {React.ReactElement} Circular toggle button with chevron icon
 */
export const CollapsibleButton = ({ isCollapsed, onClick, title }) => {
  return (
    <button
      onClick={onClick}
      style={{
        position: 'absolute',
        top: '-6px',
        right: '8px',
        width: '28px',
        height: '28px',
        borderRadius: '50%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'rgba(255, 255, 255, 0.95)',
        border: '1px solid #e5e7eb',
        color: '#9ca3af',
        cursor: 'pointer',
        fontSize: '18px',
        transition: 'all 0.2s ease',
        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
        zIndex: 10
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = '#f3f4f6';
        e.currentTarget.style.color = '#667eea';
        e.currentTarget.style.borderColor = '#d1d5db';
        e.currentTarget.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.15)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = 'rgba(255, 255, 255, 0.95)';
        e.currentTarget.style.color = '#9ca3af';
        e.currentTarget.style.borderColor = '#e5e7eb';
        e.currentTarget.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.1)';
      }}
      onMouseDown={(e) => {
        e.currentTarget.style.transform = 'scale(0.95)';
      }}
      onMouseUp={(e) => {
        e.currentTarget.style.transform = 'scale(1)';
      }}
      title={title || (isCollapsed ? 'Expand table' : 'Collapse table')}
    >
      {isCollapsed ? <HiChevronDown /> : <HiChevronUp />}
    </button>
  );
};
