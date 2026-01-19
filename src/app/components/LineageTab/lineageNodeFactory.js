/**
 * Factory functions for creating lineage graph nodes
 * Eliminates repetitive node creation code in LineageTab
 */

import { getArtifactColor } from '../../utils/artifactColors';

/**
 * Create a lineage node with standard structure
 * @param {string} type - Node type (config, code, dataset, deps, env, run, model)
 * @param {Object} data - Node-specific data
 * @param {Object} position - {x, y} position
 * @param {string} color - Optional custom color override
 * @returns {Object} ReactFlow node object
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
 */
const formatRunsList = (runs) => {
  return runs.map(r => r.name || r.run_id).join(', ');
};

/**
 * Create config node
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
