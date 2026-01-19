import React, { useEffect } from 'react';
import './TabbedInterface.scss';

/**
 * Professional tabbed interface with fixed tabs
 *
 * @param {Array} tabs - Array of tab objects with { id, label, content }
 * @param {Array} visibleTabs - Array of visible tab IDs
 * @param {string} activeTab - Currently active tab ID
 * @param {Function} onTabChange - Callback when active tab changes
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
