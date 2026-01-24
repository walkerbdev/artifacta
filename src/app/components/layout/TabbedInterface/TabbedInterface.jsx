import React, { useEffect } from 'react';
import './TabbedInterface.scss';

/**
 * Tabbed Interface component for main content area navigation
 *
 * Tab bar for switching between different analysis views (Plots, Tables, Sweeps, etc.).
 * Handles tab visibility and ensures a valid tab is always active.
 *
 * Features:
 * - Dynamic tab visibility (tabs can be shown/hidden via View menu)
 * - Active tab highlighting
 * - Auto-switches to first visible tab if active tab hidden
 * - Empty state when no tabs visible
 * - Fixed tab bar at top
 * - Content area below tabs
 *
 * Typical tabs:
 * - Plots: Auto-discovered visualizations
 * - Tables: Metric comparison tables
 * - Sweeps: Hyperparameter sweep analysis
 * - Lineage: Provenance graph
 * - Artifacts: File browser
 * - Chat: LLM experiment analysis
 * - Notes: Rich-text note taking
 *
 * @param {object} props - Component props
 * @param {Array<object>} [props.tabs=[]] - All available tabs:
 *   - id: string - Unique tab identifier
 *   - label: string - Display name
 *   - content: React.ReactNode - Tab content component
 * @param {Array<string>} [props.visibleTabs=[]] - IDs of tabs to show
 * @param {string} props.activeTab - Currently active tab ID
 * @param {function} props.onTabChange - Callback when user switches tabs
 *   Signature: (tabId: string) => void
 * @returns {React.ReactElement} Tab bar with content area
 */
export const TabbedInterface = ({
  tabs = [],
  visibleTabs = [],
  activeTab,
  onTabChange
}) => {
  // Filter to only show visible tabs
  const displayedTabs = tabs.filter(tab => visibleTabs.includes(tab.id));

  // Ensure active tab is valid
  useEffect(() => {
    if (displayedTabs.length > 0 && !displayedTabs.find(t => t.id === activeTab)) {
      onTabChange(displayedTabs[0].id);
    }
  }, [displayedTabs, activeTab, onTabChange]);

  if (displayedTabs.length === 0) {
    return (
      <div className="tabbed-interface-empty">
        <p>No panels open. Use the <strong>View</strong> menu to restore panels.</p>
      </div>
    );
  }

  return (
    <div className="tabbed-interface">
      {/* Tab Bar */}
      <div className="tab-bar">
        <div className="tab-list">
          {displayedTabs.map((tab) => (
            <button
              key={tab.id}
              className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => onTabChange(tab.id)}
              title={tab.label}
            >
              <span className="tab-label">{tab.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      <div className="tab-content-container">
        {displayedTabs.map((tab) => (
          <div
            key={tab.id}
            className="tab-content"
            style={{ display: tab.id === activeTab ? 'flex' : 'none' }}
          >
            {tab.content}
          </div>
        ))}
      </div>
    </div>
  );
};
