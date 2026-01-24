import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { HiChevronDown, HiChevronUp } from 'react-icons/hi';
import './CollapsibleSection.scss';

/**
 * Collapsible section component with animation
 * @param {object} props - Component props
 * @param {string} [props.title] - Optional section title
 * @param {React.ReactNode} props.children - Section content
 * @param {boolean} [props.defaultCollapsed] - Whether section starts collapsed
 * @returns {React.ReactElement} The collapsible section component
 */
export function CollapsibleSection({ title, children, defaultCollapsed = false }) {
  const [isCollapsed, setIsCollapsed] = useState(defaultCollapsed);

  return (
    <div className="collapsible-section-wrapper">
      {title && (
        <div className="collapsible-section-title-bar">
          <h3>{title}</h3>
        </div>
      )}
      <button
        className="collapsible-section-toggle"
        onClick={() => setIsCollapsed(!isCollapsed)}
        title={isCollapsed ? "Expand section" : "Collapse section"}
      >
        {isCollapsed ? <HiChevronDown /> : <HiChevronUp />}
      </button>
      <AnimatePresence initial={false}>
        {!isCollapsed && (
          <motion.div
            className="collapsible-section-content"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
          >
            {children}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
