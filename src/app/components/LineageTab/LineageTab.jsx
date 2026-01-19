import React, { useState, useEffect, useMemo } from 'react';
import ReactFlow, { Background, MarkerType, Handle, Position, ReactFlowProvider, useReactFlow } from 'reactflow';
import 'reactflow/dist/style.css';
import './LineageTab.scss';
import { apiClient } from '@/core/api/ApiClient';
import {
  createRunNode,
  createArtifactNode
} from './lineageNodeFactory';

// Custom compact node component
const CompactNode = ({ data }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const handleClick = (e) => {
    e.stopPropagation();
    setIsExpanded(!isExpanded);
  };


  // Use custom color if provided, otherwise CSS will handle it
  const style = data.color ? { borderColor: data.color } : {};

  return (
    <div
      className={`provenance-node ${data.type} ${isExpanded ? 'expanded' : ''}`}
      onClick={handleClick}
      style={style}
    >
      {/* Input handle (left side) */}
      <Handle type="target" position={Position.Left} />

      <div className="node-header">
        <div className="node-label">{data.label}</div>
        <div className="node-hash">{data.hashShort}</div>
      </div>

      {/* Expanded details */}
      {isExpanded && data.fullHash && (
        <div className="node-expanded">
          <div className="expanded-row">
            <span className="expanded-label">Full Hash:</span>
            <span className="expanded-value">{data.fullHash}</span>
          </div>
          {data.fullId && (
            <div className="expanded-row">
              <span className="expanded-label">ID:</span>
              <span className="expanded-value">{data.fullId}</span>
            </div>
          )}
          {data.extraInfo && Object.entries(data.extraInfo).map(([key, value]) => (
            value && (
              <div key={key} className="expanded-row">
                <span className="expanded-label">{key}:</span>
                <span className="expanded-value">{String(value)}</span>
              </div>
            )
          ))}
        </div>
      )}

      {/* Output handle (right side) */}
      <Handle type="source" position={Position.Right} />
    </div>
  );
};

const nodeTypes = {
  compact: CompactNode
};

const LineageFlow = ({ selectedRunIds, allRuns, onDatasetSelect, onArtifactView }) => {
  const [artifactLinksByRun, setArtifactLinksByRun] = useState({});
  const [loading, setLoading] = useState(false);
  const [hoveredNode, setHoveredNode] = useState(null);
  const { fitView } = useReactFlow();

  const selectedRuns = useMemo(() => {
    return selectedRunIds
      ?.map(id => allRuns.find(r => r.run_id === id))
      .filter(Boolean) || [];
  }, [selectedRunIds, allRuns]);

  // Fetch artifact links for all selected runs
  useEffect(() => {
    if (selectedRuns.length === 0) {
      setArtifactLinksByRun({});
      return;
    }

    const fetchAllArtifactLinks = async () => {
      setLoading(true);
      try {
        const results = await Promise.all(
          selectedRuns.map(async (run) => {
            try {
              const links = await apiClient.getArtifactLinks(run.run_id);
              return { runId: run.run_id, links };
            } catch (err) {
              console.error(`Failed to fetch artifact links for ${run.run_id}:`, err);
              return { runId: run.run_id, links: [] };
            }
          })
        );

        const linksMap = {};
        results.forEach(({ runId, links }) => {
          linksMap[runId] = links;
        });
        setArtifactLinksByRun(linksMap);
      } finally {
        setLoading(false);
      }
    };

    fetchAllArtifactLinks();
  }, [selectedRuns]);

  // Build graph nodes and edges
  const { nodes, edges } = useMemo(() => {
    if (selectedRuns.length === 0) return { nodes: [], edges: [] };

    const nodes = [];
    const edges = [];
    const spacing = 100;

    // Group input artifacts by hash across all runs
    const inputArtifactsByHash = {};

    // Collect all input artifacts from all runs
    selectedRuns.forEach(run => {
      const runLinks = artifactLinksByRun[run.run_id] || [];
      const inputArtifacts = runLinks.filter(link => link.role === 'input');

      inputArtifacts.forEach(link => {
        if (!inputArtifactsByHash[link.hash]) {
          inputArtifactsByHash[link.hash] = {
            artifact_id: link.artifact_id,
            name: link.name,
            hash: link.hash,
            storage_path: link.storage_path,
            size_bytes: link.size_bytes,
            metadata: link.metadata,
            content: link.content,
            runs: []
          };
        }
        if (!inputArtifactsByHash[link.hash].runs.find(r => r.run_id === run.run_id)) {
          inputArtifactsByHash[link.hash].runs.push(run);
        }
      });
    });

    // Run nodes (center, vertically stacked)
    const runYStart = 0;
    selectedRuns.forEach((run, idx) => {
      const runY = runYStart + (idx * spacing);
      nodes.push(createRunNode(run, idx, { x: 250, y: runY }));
    });

    // Sort input artifacts by their connected runs for better grouping
    const sortedInputArtifacts = Object.values(inputArtifactsByHash).sort((a, b) => {
      // Get the first run_id for each artifact (for primary sorting)
      const aFirstRunId = a.runs[0]?.run_id || '';
      const bFirstRunId = b.runs[0]?.run_id || '';

      // Find index in selectedRuns to sort by run order
      const aRunIndex = selectedRuns.findIndex(r => r.run_id === aFirstRunId);
      const bRunIndex = selectedRuns.findIndex(r => r.run_id === bFirstRunId);

      if (aRunIndex !== bRunIndex) {
        return aRunIndex - bRunIndex; // Group by run
      }

      // Within same run, sort by artifact name for consistency
      return a.name.localeCompare(b.name);
    });

    // Create input artifact nodes (left side) - now sorted by run
    let yOffset = 0;
    sortedInputArtifacts.forEach((artifactGroup) => {
      const artifactNode = createArtifactNode(
        artifactGroup,
        null,  // no specific run (shared input)
        { x: 0, y: yOffset },
        onArtifactView,
        onDatasetSelect
      );

      nodes.push(artifactNode);
      yOffset += spacing;
    });

    // Create edges and output artifacts for each run
    selectedRuns.forEach((run, idx) => {
      const runY = runYStart + (idx * spacing);

      // Create edges from input artifacts to this run
      const runLinks = artifactLinksByRun[run.run_id] || [];
      const inputArtifacts = runLinks.filter(link => link.role === 'input');
      inputArtifacts.forEach(link => {
        edges.push({
          id: `artifact-${link.hash}-${run.run_id}`,
          source: `artifact-${link.hash}`,
          target: `run-${run.run_id}`,
          type: 'smoothstep',
          animated: false,
          data: { runId: run.run_id, artifactHash: link.hash },
          markerEnd: { type: MarkerType.ArrowClosed, color: '#1e293b' }
        });
      });

      // Create edges from this run to output artifacts
      const outputArtifacts = runLinks.filter(link => link.role === 'output');

      let artifactOffset = 0;
      outputArtifacts.forEach((link) => {
        // Check if this artifact already exists as an input node (by hash)
        const existingInputArtifact = inputArtifactsByHash[link.hash];

        if (existingInputArtifact) {
          // Artifact already exists as input - just create edge to existing node
          edges.push({
            id: `${run.run_id}-artifact-${link.hash}`,
            source: `run-${run.run_id}`,
            target: `artifact-${link.hash}`,
            type: 'smoothstep',
            animated: false,
            markerEnd: { type: MarkerType.ArrowClosed, color: '#1e293b' }
          });
        } else {
          // New output artifact - create node on the right
          const artifact = {
            artifact_id: link.artifact_id,
            name: link.name,
            hash: link.hash,
            storage_path: link.storage_path,
            size_bytes: link.size_bytes,
            metadata: link.metadata,
            content: link.content
          };

          const artifactNode = createArtifactNode(
            artifact,
            run,
            { x: 500, y: runY + artifactOffset },
            onArtifactView,
            onDatasetSelect
          );

          nodes.push(artifactNode);

          edges.push({
            id: `${run.run_id}-${artifactNode.id}`,
            source: `run-${run.run_id}`,
            target: artifactNode.id,
            type: 'smoothstep',
            animated: false,
            markerEnd: { type: MarkerType.ArrowClosed, color: '#1e293b' }
          });

          artifactOffset += spacing;
        }
      });
    });

    return { nodes, edges };
  }, [selectedRuns, artifactLinksByRun, onDatasetSelect, onArtifactView]);

  // Fit view after nodes are rendered - only when node count changes
  useEffect(() => {
    if (nodes.length > 0) {
      setTimeout(() => {
        fitView({ padding: 0.1, duration: 200 });
      }, 50);
    }
  }, [nodes.length, fitView]);

  if (!selectedRunIds || selectedRunIds.length === 0) {
    return null;
  }

  if (loading) {
    return <div className="lineage-tab-loading">Loading provenance...</div>;
  }

  // Check if node is connected to hovered node
  const isNodeConnected = (nodeId, hoveredId) => {
    if (nodeId === hoveredId) return true;
    return edges.some(edge =>
      (edge.source === hoveredId && edge.target === nodeId) ||
      (edge.target === hoveredId && edge.source === nodeId)
    );
  };

  // Check if edge is connected to hovered node
  const isEdgeConnected = (edge, hoveredId) => {
    return edge.source === hoveredId || edge.target === hoveredId;
  };

  // Apply hover highlighting to nodes and edges
  const nodesWithHover = nodes.map(node => ({
    ...node,
    style: {
      ...node.style,
      opacity: hoveredNode ? (isNodeConnected(node.id, hoveredNode) ? 1 : 0.3) : 1,
      transition: 'opacity 0.2s ease'
    }
  }));

  const edgesWithHover = edges.map(edge => ({
    ...edge,
    style: {
      ...edge.style,
      opacity: hoveredNode ? (isEdgeConnected(edge, hoveredNode) ? 1 : 0.15) : 1,
      strokeWidth: hoveredNode && isEdgeConnected(edge, hoveredNode) ? 3 : 2,
      transition: 'all 0.2s ease'
    }
  }));

  const handleNodeMouseEnter = (event, node) => {
    setHoveredNode(node.id);
  };

  const handleNodeMouseLeave = () => {
    setHoveredNode(null);
  };

  return (
    <ReactFlow
      nodes={nodesWithHover}
      edges={edgesWithHover}
      nodeTypes={nodeTypes}
      minZoom={0.5}
      maxZoom={2}
      nodesDraggable={true}
      nodesConnectable={false}
      elementsSelectable={true}
      onNodeMouseEnter={handleNodeMouseEnter}
      onNodeMouseLeave={handleNodeMouseLeave}
      proOptions={{ hideAttribution: true }}
      style={{ width: '100%', height: '100%' }}
    >
      <Background color="#f1f5f9" gap={16} />
    </ReactFlow>
  );
};

export const LineageTab = (props) => {
  return (
    <ReactFlowProvider>
      <div className="lineage-tab">
        <LineageFlow {...props} />
      </div>
    </ReactFlowProvider>
  );
};
