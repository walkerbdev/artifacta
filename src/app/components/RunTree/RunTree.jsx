import React, { useState, useMemo } from 'react';
import { SIDEBAR_STYLES } from '@/app/styles/sidebarConstants';

/**
 * Hierarchy definition for run tree grouping
 *
 * Defines the order of hash-based grouping from most to least important.
 * Runs are grouped by code changes first, then config, then environment, etc.
 * This creates a deterministic tree structure for reproducibility tracking.
 */
const HASH_HIERARCHY = [
  { key: 'hash.code', label: 'Code' },
  { key: 'hash.config', label: 'Config' },
  { key: 'hash.environment', label: 'Env' },
  { key: 'hash.dependencies', label: 'Deps' },
  { key: 'hash.platform', label: 'Platform' }
];

/**
 * Recursive tree builder with auto-collapsing of redundant levels
 *
 * Groups runs hierarchically by hash values in the specified order.
 * Automatically skips levels where all runs have identical values (no point grouping).
 *
 * Algorithm:
 * 1. At each level, group runs by current hash key value
 * 2. If only one unique value exists, skip this level (redundant)
 * 3. Otherwise, create nodes for each unique value
 * 4. Recursively build children at next hierarchy level
 *
 * @param {Array<object>} runs - Runs to organize
 * @param {Array<object>} [hierarchy=HASH_HIERARCHY] - Hash keys to group by
 * @param {number} [depth=0] - Current recursion depth
 * @returns {Array} Tree nodes: { hashKey, hashValue, label, runs, children }
 */
function buildHashTree(runs, hierarchy = HASH_HIERARCHY, depth = 0) {
  // Base case: no more hierarchy levels or no runs
  if (depth >= hierarchy.length || runs.length === 0) {
    return runs;
  }

  const currentLevel = hierarchy[depth];
  const grouped = {};

  // Group runs by current hash key
  runs.forEach(run => {
    const hashValue = run.tags?.[currentLevel.key] || null;
    const groupKey = hashValue || 'unknown';

    if (!grouped[groupKey]) {
      grouped[groupKey] = [];
    }
    grouped[groupKey].push(run);
  });

  // Check if this level creates a meaningful split
  const uniqueValues = Object.keys(grouped);

  // If only 1 unique value at this level AND not at root, skip it (redundant grouping)
  // Always show at least the root level even if there's only 1 value
  if (uniqueValues.length === 1 && depth > 0) {
    // Skip this level and recurse to next
    return buildHashTree(runs, hierarchy, depth + 1);
  }

  // Recursively build children for each group
  const nodes = Object.entries(grouped).map(([hashValue, groupRuns]) => {
    // Sort runs within group by timestamp (newest first)
    groupRuns.sort((a, b) => b.run_id.localeCompare(a.run_id));

    return {
      hashKey: currentLevel.key,
      hashValue: hashValue === 'unknown' ? null : hashValue,
      label: currentLevel.label,
      formatValue: currentLevel.formatValue,
      runs: groupRuns,
      newestRunId: groupRuns[0].run_id,
      depth,
      // Recursively build children at next level
      children: buildHashTree(groupRuns, hierarchy, depth + 1)
    };
  });

  // Sort nodes by newest run
  nodes.sort((a, b) => b.newestRunId.localeCompare(a.newestRunId));

  return nodes;
}


/**
 * Recursive tree node renderer
 * @param {object} props - Component props
 * @param {object} props.node - Tree node data
 * @param {Array} props.allRuns - All runs in the tree
 * @param {Array<string>} props.selectedRunIds - Array of selected run IDs
 * @param {(runIds: Array<string>) => void} props.onRunSelectionChange - Callback for selection changes
 * @param {Set} props.collapsedNodes - Set of collapsed node IDs
 * @param {(nodeId: string) => void} props.toggleNode - Callback to toggle node collapse state
 * @returns {React.ReactElement} Rendered tree node
 */
function TreeNode({ node, allRuns, selectedRunIds, onRunSelectionChange, collapsedNodes, toggleNode }) {

  const isLeaf = Array.isArray(node.children) && node.children.every(child => child.run_id);
  const nodeId = `${node.hashKey}:${node.hashValue}`;
  const isCollapsed = collapsedNodes.has(nodeId);
  const isNodeSelected = node.runs?.some(r => selectedRunIds.includes(r.run_id)) ?? false;

  /**
   * Handles click event on a run item to toggle its selection
   * @param {string} runId - ID of the run to toggle
   * @returns {void}
   */
  const handleRunClick = (runId) => {
    const isSelected = selectedRunIds.includes(runId);
    const newSelection = isSelected
      ? selectedRunIds.filter(id => id !== runId)
      : [...selectedRunIds, runId];
    onRunSelectionChange(newSelection);
  };

  // Format display value
  const displayValue = (() => {
    if (!node.hashValue) return null; // Hide "unknown" groupings
    if (node.formatValue) return node.formatValue(node.hashValue);
    return node.hashValue.slice(0, 8);
  })();

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
      {/* Node Header */}
      <div
        onClick={() => toggleNode(nodeId)}
        style={{
          padding: SIDEBAR_STYLES.spacing.padding,
          fontSize: SIDEBAR_STYLES.fontSize.header,
          color: 'black',
          cursor: 'pointer',
          userSelect: 'none',
          display: 'flex',
          alignItems: 'center',
          gap: SIDEBAR_STYLES.spacing.gap,
          background: isNodeSelected ? 'white' : 'transparent',
          borderRadius: '4px',
          transition: 'background 0.2s'
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.background = 'white';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.background = isNodeSelected ? 'white' : 'transparent';
        }}
      >
        <span style={{ fontSize: '10px', color: 'black' }}>
          {isCollapsed ? '▶' : '▼'}
        </span>
        <span style={{ fontSize: '0.65rem', color: 'black', textTransform: 'uppercase' }}>
          {node.label}:
        </span>
        {displayValue && (
          <span style={{
            fontFamily: SIDEBAR_STYLES.fontFamily,
            fontSize: SIDEBAR_STYLES.fontSize.header,
            fontWeight: SIDEBAR_STYLES.fontWeight.header
          }}>
            {displayValue}
          </span>
        )}
      </div>

      {/* Children (either tree nodes or leaf runs) */}
      {!isCollapsed && (
        <div style={{ paddingLeft: '16px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
          {isLeaf ? (
            // Leaf nodes: render runs
            node.children.map((run) => {
              const isSelected = selectedRunIds.includes(run.run_id);

              return (
                <div
                  key={run.run_id}
                  onClick={() => handleRunClick(run.run_id)}
                  style={{
                    padding: '8px',
                    fontSize: SIDEBAR_STYLES.fontSize.item,
                    borderRadius: '4px',
                    background: 'white',
                    color: 'black',
                    cursor: 'pointer',
                    transition: 'all 0.2s',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = 'white';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'white';
                  }}
                >
                  {/* Checkbox for selection */}
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => handleRunClick(run.run_id)}
                    onClick={(e) => e.stopPropagation()}
                    style={{
                      flexShrink: 0,
                      cursor: 'pointer'
                    }}
                  />

                  {/* Run name */}
                  <span style={{
                    flex: 1,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    fontFamily: SIDEBAR_STYLES.fontFamily,
                    fontWeight: SIDEBAR_STYLES.fontWeight.item
                  }}>
                    {run.name || run.run_id}
                  </span>
                </div>
              );
            })
          ) : (
            // Internal nodes: render child tree nodes recursively
            node.children?.map((childNode, idx) => (
              <TreeNode
                key={`${childNode.hashKey}:${childNode.hashValue}:${idx}`}
                node={childNode}
                allRuns={allRuns}
                selectedRunIds={selectedRunIds}
                onRunSelectionChange={onRunSelectionChange}
                collapsedNodes={collapsedNodes}
                toggleNode={toggleNode}
              />
            )) ?? null
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Run Tree component for hierarchical run organization and selection
 *
 * Git-style tree browser that groups experiment runs by reproducibility hashes.
 * Helps identify which runs were conducted under identical conditions (code, config,
 * environment, dependencies, platform).
 *
 * Features:
 * - Multi-level hash-based grouping (code → config → env → deps → platform)
 * - Auto-collapse redundant levels (skips levels where all runs are identical)
 * - Collapsible tree nodes
 * - Multi-select with checkboxes (supports selecting entire groups)
 * - Visual hierarchy with indentation
 * - Shows run counts per group
 * - Newest-first sorting
 *
 * Use cases:
 * - Finding reproducible runs (same code + config = should get same results)
 * - Identifying what changed between runs
 * - Bulk-selecting runs from same experiment batch
 * - Debugging environment/dependency issues
 *
 * Hash hierarchy (top to bottom):
 * 1. Code - Git commit hash or code content hash
 * 2. Config - Hyperparameter configuration hash
 * 3. Environment - Environment variables hash
 * 4. Dependencies - Package versions hash
 * 5. Platform - OS/hardware hash
 *
 * @param {object} props - Component props
 * @param {Array<object>} props.runs - Experiment runs to organize
 * @param {Array<string>} props.selectedRunIds - Currently selected run IDs
 * @param {function} props.onRunSelectionChange - Callback when selection changes
 *   Signature: (runIds: Array<string>) => void
 * @returns {React.ReactElement} Hierarchical tree with selectable runs
 */
export function RunTree({ runs, selectedRunIds, onRunSelectionChange }) {
  // Track which nodes are collapsed (by nodeId: "hashKey:hashValue")
  const [collapsedNodes, setCollapsedNodes] = useState(new Set());

  // Build tree only when run list changes (not when metrics update)
  const treeNodes = useMemo(() => {
    const result = buildHashTree(runs);
    return result;
  }, [runs]);

  /**
   * Toggles the collapsed state of a tree node
   * @param {string} nodeId - ID of the node to toggle
   * @returns {void}
   */
  const toggleNode = (nodeId) => {
    setCollapsedNodes(prev => {
      const next = new Set(prev);
      if (next.has(nodeId)) {
        next.delete(nodeId);
      } else {
        next.add(nodeId);
      }
      return next;
    });
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
      {treeNodes.map((node, idx) => (
        <TreeNode
          key={`${node.hashKey}:${node.hashValue}:${idx}`}
          node={node}
          allRuns={runs}
          selectedRunIds={selectedRunIds}
          onRunSelectionChange={onRunSelectionChange}
          collapsedNodes={collapsedNodes}
          toggleNode={toggleNode}
        />
      ))}
    </div>
  );
}
