/**
 * Factory functions for creating lineage graph nodes
 *
 * Provides standardized node creation for ReactFlow lineage visualization.
 * Eliminates repetitive node setup code and ensures consistent node structure
 * across all node types (runs, artifacts, configs, code, environment, etc.).
 *
 * Node types supported:
 * - run: Experiment run nodes
 * - artifact: Generic file artifacts (logs, images, etc.)
 * - model: Model checkpoint artifacts
 * - dataset: Dataset artifacts
 * - config: Configuration hash nodes
 * - code: Code hash nodes (git commits)
 * - env: Environment variable hash nodes
 * - deps: Dependency hash nodes
 *
 * All nodes share common structure:
 * - Compact display mode by default
 * - Expandable on click to show full details
 * - Hash-based deduplication
 * - Color coding for artifacts
 * - Click handlers for viewing/selecting
 */

import { getArtifactColor } from '../../utils/artifactColors';

/**
 * Create a lineage node with standard structure
 * @param {string} type - Node type (config, code, dataset, deps, env, run, model)
 * @param {object} data - Node-specific data
 * @param {object} position - {x, y} position
 * @param {string} color - Optional custom color override
 * @returns {object} ReactFlow node object
 */
const createLineageNode = (type, data, position, color) => {
  const { id, label, hash, extraInfo = {}, ...rest } = data;

  return {
    id,
    type: 'compact',
    position,
    data: {
      label,
      type,
      hashShort: hash ? hash.substring(0, 8) : undefined,
      fullHash: hash,
      extraInfo,
      color, // Custom color for artifact nodes
      ...rest
    }
  };
};

/**
 * Format runs list for node display
 * @param {Array} runs - Array of run objects
 * @returns {string} Comma-separated list of run names/IDs
 */
const formatRunsList = (runs) => {
  return runs.map(r => r.name || r.run_id).join(', ');
};

/**
 * Create config node
 * @param {object} configGroup - Configuration group data
 * @param {object} position - Node position {x, y}
 * @param {(artifact: object) => void} onView - Handler to view config JSON
 * @param {object} virtualArtifact - Virtual artifact with JSON content
 * @returns {object} ReactFlow node object
 */
export const createConfigNode = (configGroup, position, onView, virtualArtifact) => {
  return createLineageNode('config', {
    id: `config-${configGroup.hash}`,
    label: `Config (${configGroup.runs.length})`,
    hash: configGroup.hash,
    onArtifactView: onView,  // Handler to view config JSON
    artifactData: virtualArtifact,  // Virtual artifact with JSON content
    extraInfo: {
      'Runs': formatRunsList(configGroup.runs)
    }
  }, position);
};

/**
 * Create code node
 * @param {object} codeGroup - Code group data
 * @param {object} position - Node position {x, y}
 * @returns {object} ReactFlow node object
 */
export const createCodeNode = (codeGroup, position) => {
  return createLineageNode('code', {
    id: `code-${codeGroup.hash}`,
    label: `Code (${codeGroup.runs.length})`,
    hash: codeGroup.hash,
    extraInfo: {
      'Git Commit': codeGroup.gitCommit,
      'Git Branch': codeGroup.gitBranch,
      'Runs': formatRunsList(codeGroup.runs)
    }
  }, position);
};

/**
 * Create dataset node
 * @param {object} datasetGroup - Dataset group data
 * @param {object} position - Node position {x, y}
 * @param {(dataset: object) => void} onDatasetSelect - Handler for dataset selection
 * @returns {object} ReactFlow node object
 */
export const createDatasetNode = (datasetGroup, position, onDatasetSelect) => {
  return createLineageNode('dataset', {
    id: `artifact-${datasetGroup.hash}`,
    label: `${datasetGroup.name} (${datasetGroup.runs.length})`,
    hash: datasetGroup.hash,
    fullId: datasetGroup.artifact_id,
    artifactId: datasetGroup.artifact_id,
    datasetData: datasetGroup,
    onDatasetSelect,
    extraInfo: {
      'Path': datasetGroup.storage_path,
      'Runs': formatRunsList(datasetGroup.runs)
    }
  }, position);
};

/**
 * Create dependencies node
 * @param {object} depsGroup - Dependencies group data
 * @param {object} position - Node position {x, y}
 * @returns {object} ReactFlow node object
 */
export const createDepsNode = (depsGroup, position) => {
  return createLineageNode('deps', {
    id: `deps-${depsGroup.hash}`,
    label: `Deps (${depsGroup.runs.length})`,
    hash: depsGroup.hash,
    extraInfo: {
      'Runs': formatRunsList(depsGroup.runs)
    }
  }, position);
};

/**
 * Create environment node
 * @param {object} envGroup - Environment group data
 * @param {object} position - Node position {x, y}
 * @returns {object} ReactFlow node object
 */
export const createEnvNode = (envGroup, position) => {
  return createLineageNode('env', {
    id: `env-${envGroup.hash}`,
    label: `Env (${envGroup.runs.length})`,
    hash: envGroup.hash,
    extraInfo: {
      'Runs': formatRunsList(envGroup.runs)
    }
  }, position);
};

/**
 * Create run node
 * @param {object} run - Run data
 * @param {number} idx - Run index
 * @param {object} position - Node position {x, y}
 * @returns {object} ReactFlow node object
 */
export const createRunNode = (run, idx, position) => {
  const runLabel = run.name || `Run ${idx + 1}`;

  return createLineageNode('run', {
    id: `run-${run.run_id}`,
    label: runLabel,
    hash: run.run_id,
    fullId: run.run_id,
    extraInfo: {
      'Run ID': run.run_id,
      'Sweep ID': run.sweep_id,
      'Name': run.name,
      'Status': run.status,
      'Created': run.start_time ? new Date(run.start_time).toLocaleString() : 'N/A'
    }
  }, position);
};

/**
 * Create model node
 * @param {object} model - Model artifact data
 * @param {object} run - Run data (null for input artifacts)
 * @param {object} position - Node position {x, y}
 * @param {(artifact: object) => void} onArtifactView - Handler to view artifact
 * @returns {object} ReactFlow node object
 */
export const createModelNode = (model, run, position, onArtifactView) => {
  // If run is null, this is an input artifact (grouped by hash)
  const isInput = !run;

  return createLineageNode('model', {
    id: isInput ? `artifact-${model.hash}` : `model-${run.run_id}-${model.artifact_id}`,
    label: isInput ? `${model.name} (${model.runs.length})` : model.name,
    hash: model.hash,
    fullId: model.artifact_id,
    artifactId: model.artifact_id,
    artifactData: model,
    onArtifactView,
    extraInfo: isInput ? {
      'Path': model.storage_path,
      'Runs': formatRunsList(model.runs)
    } : {
      'Path': model.storage_path,
      'Run': run.name || run.run_id
    }
  }, position);
};

/**
 * Create generic artifact node
 * Handles both inputs (run=null) and outputs (run provided)
 * Color-coded by file extension
 * @param {object} artifact - Artifact data
 * @param {object} run - Run data (null for input artifacts)
 * @param {object} position - Node position {x, y}
 * @param {(artifact: object) => void} onArtifactView - Handler to view artifact
 * @param {(dataset: object) => void} onDatasetSelect - Handler for dataset selection
 * @returns {object} ReactFlow node object
 */
export const createArtifactNode = (artifact, run, position, onArtifactView, onDatasetSelect) => {
  // If run is null, this is an input artifact (grouped by hash)
  const isInput = !run;

  // Get color based on artifact name (file extension)
  const color = getArtifactColor(artifact.name);

  const extraInfo = isInput ? {
    'Path': artifact.storage_path,
    'Runs': formatRunsList(artifact.runs)
  } : {
    'Path': artifact.storage_path,
    'Size': artifact.size_bytes ? `${(artifact.size_bytes / 1024).toFixed(1)} KB` : 'N/A',
    'Run': run.name || run.run_id
  };

  return createLineageNode('artifact', {
    id: isInput ? `artifact-${artifact.hash}` : `artifact-${run.run_id}-${artifact.artifact_id}`,
    label: isInput ? `${artifact.name} (${artifact.runs.length})` : artifact.name,
    hash: artifact.hash,
    fullId: artifact.artifact_id,
    artifactId: artifact.artifact_id,
    artifactData: artifact,
    onArtifactView,
    onDatasetSelect,
    extraInfo
  }, position, color);
};
