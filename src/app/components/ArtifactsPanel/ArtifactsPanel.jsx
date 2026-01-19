import React, { useState, useEffect } from 'react';
import {
  SiPython, SiJavascript, SiTypescript, SiRust, SiGo, SiC, SiCplusplus,
  SiJson, SiMarkdown
} from 'react-icons/si';
import {
  VscFile, VscFilePdf, VscFileMedia, VscDatabase, VscCode,
  VscFileZip, VscGraph, VscFolder
} from 'react-icons/vsc';
import { AiOutlineFileText, AiOutlineTable } from 'react-icons/ai';
import './ArtifactsPanel.scss';

/**
 * Build a tree structure from flat file paths
 * @param {Array} files - Array of file objects with path property
 * @returns {Object} Tree structure { folders: {}, files: [] }
 */
const buildFileTree = (files) => {
  const tree = { folders: {}, files: [] };

  files.forEach((file) => {
    const parts = file.path.split('/');

    if (parts.length === 1) {
      // Root level file
      tree.files.push(file);
    } else {
      // File in folder(s)
      let currentLevel = tree;

      // Navigate/create folder structure
      for (let i = 0; i < parts.length - 1; i++) {
        const folderName = parts[i];
        if (!currentLevel.folders[folderName]) {
          currentLevel.folders[folderName] = { folders: {}, files: [] };
        }
        currentLevel = currentLevel.folders[folderName];
      }

      // Add file to deepest folder
      currentLevel.files.push({ ...file, displayName: parts[parts.length - 1] });
    }
  });

  return tree;
};

/**
 * FilesPanel - Shows files for selected runs in a collapsible folder tree
 * Displays file collections with ability to expand/collapse folders and view individual files
 */
export const ArtifactsPanel = ({ selectedRunIds, onFileSelect }) => {
  const [artifacts, setArtifacts] = useState([]);
  const [expandedArtifacts, setExpandedArtifacts] = useState(new Set());
  const [expandedFolders, setExpandedFolders] = useState(new Set());
  const [loading, setLoading] = useState(false);

  // Fetch artifacts for selected runs
  useEffect(() => {
    if (!selectedRunIds || selectedRunIds.length === 0) {
      setArtifacts([]);
      return;
    }

    const fetchArtifacts = async () => {
      setLoading(true);
      try {
        // Fetch artifact links for all selected runs
        const allArtifacts = [];
        const runNames = {};

        for (const runId of selectedRunIds) {
          const response = await fetch(`${import.meta.env.VITE_API_URL}/api/runs/${runId}/artifact-links`);
          if (response.ok) {
            const links = await response.json();
            allArtifacts.push(...links.map(link => ({ ...link, run_id: runId })));
          }

          // Fetch run name for display
          const runResponse = await fetch(`${import.meta.env.VITE_API_URL}/api/runs/${runId}`);
          if (runResponse.ok) {
            const runData = await runResponse.json();
            runNames[runId] = runData.name || runId.substring(0, 8);
          }
        }

        // Group by artifact hash (same content = same artifact across runs)
        const artifactGroups = {};
        allArtifacts.forEach(artifact => {
          const key = artifact.hash;
          if (!artifactGroups[key]) {
            artifactGroups[key] = {
              ...artifact,
              run_ids: [artifact.run_id],
              run_names: [runNames[artifact.run_id]]
            };
          } else {
            artifactGroups[key].run_ids.push(artifact.run_id);
            artifactGroups[key].run_names.push(runNames[artifact.run_id]);
          }
        });

        // Sort artifacts by run and role
        const sortedArtifacts = Object.values(artifactGroups).sort((a, b) => {
          // Get first run_id for each artifact
          const aFirstRunId = a.run_ids[0] || '';
          const bFirstRunId = b.run_ids[0] || '';

          // Find index in selectedRunIds to sort by run order
          const aRunIndex = selectedRunIds.indexOf(aFirstRunId);
          const bRunIndex = selectedRunIds.indexOf(bFirstRunId);

          if (aRunIndex !== bRunIndex) {
            return aRunIndex - bRunIndex; // Group by run
          }

          // Within same run, sort by role (input before output)
          const roleOrder = { input: 0, output: 1 };
          const aRoleOrder = roleOrder[a.role] ?? 2;
          const bRoleOrder = roleOrder[b.role] ?? 2;

          if (aRoleOrder !== bRoleOrder) {
            return aRoleOrder - bRoleOrder;
          }

          // Finally, sort by artifact name for consistency
          return a.name.localeCompare(b.name);
        });

        setArtifacts(sortedArtifacts);
      } catch (error) {
        console.error('Failed to fetch artifacts:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchArtifacts();
  }, [selectedRunIds]);

  const toggleArtifact = (artifactId) => {
    setExpandedArtifacts(prev => {
      const next = new Set(prev);
      if (next.has(artifactId)) {
        next.delete(artifactId);
      } else {
        next.add(artifactId);
      }
      return next;
    });
  };

  // Helper to get a specific folder's tree from the full tree
  const getFolderTree = (tree, path) => {
    const parts = path.split('/');
    let current = tree;
    for (const part of parts) {
      if (!current.folders[part]) return null;
      current = current.folders[part];
    }
    return current;
  };

  // Generic helper to check if folder contains only files of a specific media type
  const isFolderOnlyMediaType = (folderTree, mimePrefix) => {
    if (Object.keys(folderTree.folders).length > 0) return false;
    return folderTree.files.every(file => file.mime_type?.startsWith(mimePrefix));
  };

  // Generic helper to collect files by media type (recursively)
  const collectFilesByMediaType = (folderTree, mimePrefix) => {
    const files = [];
    folderTree.files.forEach(file => {
      if (file.mime_type?.startsWith(mimePrefix)) {
        files.push(file);
      }
    });
    Object.entries(folderTree.folders).forEach(([, subTree]) => {
      files.push(...collectFilesByMediaType(subTree, mimePrefix));
    });
    return files;
  };

  // Convenience wrappers for specific media types
  const isFolderOnlyImages = (folderTree) => isFolderOnlyMediaType(folderTree, 'image/');
  const isFolderOnlyAudio = (folderTree) => isFolderOnlyMediaType(folderTree, 'audio/');
  const isFolderOnlyVideo = (folderTree) => isFolderOnlyMediaType(folderTree, 'video/');

  const collectImageFiles = (folderTree) => collectFilesByMediaType(folderTree, 'image/');
  const collectAudioFiles = (folderTree) => collectFilesByMediaType(folderTree, 'audio/');
  const collectVideoFiles = (folderTree) => collectFilesByMediaType(folderTree, 'video/');

  const toggleFolder = (folderPath, fullTree, artifact) => {
    const folderTree = getFolderTree(fullTree, folderPath);

    if (!folderTree || !onFileSelect) {
      // Normal folder toggle if no special handling needed
      setExpandedFolders(prev => {
        const next = new Set(prev);
        if (next.has(folderPath)) {
          next.delete(folderPath);
        } else {
          next.add(folderPath);
        }
        return next;
      });
      return;
    }

    // Media type configurations: [checkFn, collectFn, filesKey]
    const mediaTypes = [
      [isFolderOnlyImages, collectImageFiles, 'imageFiles'],
      [isFolderOnlyAudio, collectAudioFiles, 'audioFiles'],
      [isFolderOnlyVideo, collectVideoFiles, 'videoFiles']
    ];

    // Check each media type and open grid view if matched
    for (const [checkFn, collectFn, filesKey] of mediaTypes) {
      if (checkFn(folderTree)) {
        const files = collectFn(folderTree);
        if (files.length > 0) {
          onFileSelect({
            artifact_id: artifact.artifact_id,
            name: artifact.name,
            folder: folderPath,
            [filesKey]: files,
            allFiles: parseFileCollection(artifact.content),
            metadata: artifact.metadata,
            hash: artifact.hash,
            size_bytes: artifact.size_bytes
          });
          break;
        }
      }
    }

    // Always toggle folder expansion
    setExpandedFolders(prev => {
      const next = new Set(prev);
      if (next.has(folderPath)) {
        next.delete(folderPath);
      } else {
        next.add(folderPath);
      }
      return next;
    });
  };

  const parseFileCollection = (content) => {
    if (!content) return null;
    try {
      return JSON.parse(content);
    } catch {
      return null;
    }
  };

  const getFileIcon = (file) => {
    const ext = file.path.split('.').pop()?.toLowerCase();
    const mimeType = file.mime_type;

    // Code files with language-specific icons
    if (file.metadata?.language === 'python' || ext === 'py') return <SiPython />;
    if (file.metadata?.language === 'javascript' || ext === 'js') return <SiJavascript />;
    if (file.metadata?.language === 'typescript' || ext === 'ts') return <SiTypescript />;
    if (ext === 'rs') return <SiRust />;
    if (ext === 'go') return <SiGo />;
    if (ext === 'c' || ext === 'h') return <SiC />;
    if (ext === 'cpp' || ext === 'cc' || ext === 'cxx') return <SiCplusplus />;

    // Config/Data files
    if (ext === 'json') return <SiJson />;
    if (ext === 'yaml' || ext === 'yml') return <VscFile />;
    if (ext === 'csv') return <AiOutlineTable />;
    if (ext === 'sql') return <VscDatabase />;

    // Documents
    if (file.metadata?.language === 'markdown' || ext === 'md') return <SiMarkdown />;
    if (ext === 'txt' || ext === 'log') return <AiOutlineFileText />;
    if (ext === 'pdf') return <VscFilePdf />;

    // Images
    if (mimeType?.startsWith('image/')) return <VscFileMedia />;

    // Models/Data
    if (ext === 'pt' || ext === 'pth' || ext === 'ckpt' || ext === 'h5' || ext === 'hdf5' ||
        ext === 'pkl' || ext === 'pickle' || ext === 'npz' || ext === 'npy') {
      return <VscGraph />;
    }

    // Archives
    if (ext === 'zip' || ext === 'tar' || ext === 'gz') return <VscFileZip />;

    // Generic code file
    if (file.metadata?.language) return <VscCode />;

    // Default
    return <VscFile />;
  };

  const handleFileClick = (artifact, file) => {
    if (onFileSelect) {
      onFileSelect({
        artifact_id: artifact.artifact_id,
        name: artifact.name,
        file,
        allFiles: parseFileCollection(artifact.content),
        metadata: artifact.metadata,  // Pass metadata from artifact-links
        hash: artifact.hash,
        size_bytes: artifact.size_bytes
      });
    }
  };

  // Recursive component to render file tree
  const FileTreeNode = ({ tree, artifact, pathPrefix = '', fullTree = null }) => {
    if (!tree) return null;
    const rootTree = fullTree || tree;

    return (
      <>
        {/* Render folders */}
        {Object.entries(tree.folders).map(([folderName, subTree]) => {
          const folderPath = pathPrefix ? `${pathPrefix}/${folderName}` : folderName;
          const isFolderExpanded = expandedFolders.has(folderPath);

          return (
            <div key={folderPath} className="folder-item">
              <div
                className="folder-header"
                onClick={() => toggleFolder(folderPath, rootTree, artifact)}
              >
                <span className="folder-expand-icon">
                  {isFolderExpanded ? '▼' : '▶'}
                </span>
                <span className="folder-icon"><VscFolder /></span>
                <span className="folder-name">{folderName}/</span>
              </div>
              {isFolderExpanded && (
                <div className="folder-contents">
                  <FileTreeNode tree={subTree} artifact={artifact} pathPrefix={folderPath} fullTree={rootTree} />
                </div>
              )}
            </div>
          );
        })}

        {/* Render files */}
        {tree.files.map((file, idx) => (
          <div
            key={`${pathPrefix}-${idx}`}
            className="file-item"
            onClick={() => handleFileClick(artifact, file)}
          >
            <span className="file-icon">{getFileIcon(file)}</span>
            <span className="file-path">{file.displayName || file.path}</span>
          </div>
        ))}
      </>
    );
  };

  if (loading) {
    return <div className="artifacts-panel-loading">Loading files...</div>;
  }

  if (artifacts.length === 0) {
    return null;
  }

  return (
    <div className="artifacts-panel">
      {artifacts.map(artifact => {
        const fileCollection = parseFileCollection(artifact.content);
        const isExpanded = expandedArtifacts.has(artifact.artifact_id);
        const fileCount = fileCollection?.files?.length || 0;
        const isSingleFile = fileCount === 1;
        const runCount = artifact.run_names?.length || 1;

        // Check if artifact contains only images
        const isImageOnlyArtifact = fileCollection?.files?.every(file =>
          file.mime_type?.startsWith('image/')
        );

        // Check if artifact contains only audio
        const isAudioOnlyArtifact = fileCollection?.files?.every(file =>
          file.mime_type?.startsWith('audio/')
        );

        // Check if artifact contains only video
        const isVideoOnlyArtifact = fileCollection?.files?.every(file =>
          file.mime_type?.startsWith('video/')
        );

        // Handle clicks on the expand icon
        const handleExpandClick = (e) => {
          e.stopPropagation(); // Prevent triggering header click
          toggleArtifact(artifact.artifact_id);
        };

        // For single file, click opens it directly
        // For image/audio/video-only artifacts, click opens grid view AND toggles expansion
        const handleHeaderClick = () => {
          if (isSingleFile && fileCollection) {
            handleFileClick(artifact, fileCollection.files[0]);
            return;
          }

          if (fileCollection && onFileSelect) {
            // Media type configurations: [checkFlag, filesKey]
            const mediaTypes = [
              [isImageOnlyArtifact, 'imageFiles'],
              [isAudioOnlyArtifact, 'audioFiles'],
              [isVideoOnlyArtifact, 'videoFiles']
            ];

            // Check each media type and open grid view if matched
            for (const [isMediaType, filesKey] of mediaTypes) {
              if (isMediaType) {
                onFileSelect({
                  artifact_id: artifact.artifact_id,
                  name: artifact.name,
                  folder: artifact.name,
                  [filesKey]: fileCollection.files,
                  allFiles: fileCollection,
                  metadata: artifact.metadata,
                  hash: artifact.hash,
                  size_bytes: artifact.size_bytes
                });
                break;
              }
            }
          }

          // Always toggle artifact expansion (unless single file)
          toggleArtifact(artifact.artifact_id);
        };

        const tooltipText = artifact.run_names?.length
          ? `Used by: ${artifact.run_names.join(', ')}`
          : 'No run information';

        return (
          <div key={artifact.artifact_id} className="artifact-item" title={tooltipText}>
            <div
              className="artifact-header"
              onClick={handleHeaderClick}
            >
              {!isSingleFile && (
                <span className="artifact-expand-icon" onClick={handleExpandClick}>
                  {isExpanded ? '▼' : '▶'}
                </span>
              )}
              <div className="artifact-title">
                <span className="artifact-name">
                  {artifact.name}
                </span>
                {runCount > 1 && (
                  <span className="artifact-run-count">[{runCount} runs]</span>
                )}
              </div>
            </div>

            {isExpanded && fileCollection && !isSingleFile && (
              <div className="artifact-files">
                <FileTreeNode
                  tree={buildFileTree(fileCollection.files)}
                  artifact={artifact}
                />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};
