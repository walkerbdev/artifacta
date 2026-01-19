import React, { useState, useMemo } from 'react';
import { SIDEBAR_STYLES } from '@/app/styles/sidebarConstants';

/**
 * Hierarchy definition: order of hash keys for grouping (most → least important)
 * This defines the tree structure from root to leaves
 */
const HASH_HIERARCHY = [
  { key: 'hash.code', label: 'Code' },
  { key: 'hash.config', label: 'Config' },
  { key: 'hash.environment', label: 'Env' },
  { key: 'hash.dependencies', label: 'Deps' },
  { key: 'hash.platform', label: 'Platform' }
];

/**
 * Generic recursive tree builder with auto-collapsing of redundant levels
 * Groups runs hierarchically by hash keys in specified order
 * Skips levels where all runs have the same value (redundant grouping)
 *
 * @param {Array} runs - Array of run objects
 * @param {Array} hierarchy - Array of hash keys to group by (in order)
 * @param {number} depth - Current depth in tree (0 = root)
 * @returns {Array} Tree nodes with structure: { hashKey, hashValue, label, runs, children }
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
 */
function TreeNode({ node, allRuns, selectedRunIds, onRunSelectionChange, collapsedNodes, toggleNode }) {

  const isLeaf = Array.isArray(node.children) && node.children.every(child => child.run_id);
  const nodeId = `${node.hashKey}:${node.hashValue}`;
  const isCollapsed = collapsedNodes.has(nodeId);
  const isNodeSelected = node.runs?.some(r => selectedRunIds.includes(r.run_id)) ?? false;

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
 * Git-tree style run browser with multi-level hash-based grouping
 */
export function RunTree({ runs, selectedRunIds, onRunSelectionChange }) {
  // Track which nodes are collapsed (by nodeId: "hashKey:hashValue")
  const [collapsedNodes, setCollapsedNodes] = useState(new Set());

  // Build tree only when run list changes (not when metrics update)
  const treeNodes = useMemo(() => {
    const result = buildHashTree(runs);
    return result;
  }, [runs]);

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
