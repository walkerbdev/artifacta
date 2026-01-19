import React, { useState, useEffect } from 'react';
import Papa from 'papaparse';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { prism } from 'react-syntax-highlighter/dist/esm/styles/prism';
import './ArtifactTab.scss';
import { apiClient } from '@/core/api/ApiClient';

const API_BASE_URL = import.meta.env.VITE_API_URL;

export const ArtifactTab = ({ selectedArtifact }) => {
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [offset, setOffset] = useState(0);
  const [limit] = useState(100);

  useEffect(() => {
    if (!selectedArtifact) {
      setPreview(null);
      return;
    }

    // Handle image folder (clicked folder with only images from Files panel)
    if (selectedArtifact.imageFiles && selectedArtifact.imageFiles.length > 0) {
      setPreview({
        type: 'image',
        files: selectedArtifact.imageFiles,
        artifact_id: selectedArtifact.artifact_id
      });
      setLoading(false);
      return;
    }

    // Handle audio folder (clicked folder with only audio files from Files panel)
    if (selectedArtifact.audioFiles && selectedArtifact.audioFiles.length > 0) {
      setPreview({
        type: 'audio',
        files: selectedArtifact.audioFiles,
        artifact_id: selectedArtifact.artifact_id
      });
      setLoading(false);
      return;
    }

    // Handle video folder (clicked folder with only video files from Files panel)
    if (selectedArtifact.videoFiles && selectedArtifact.videoFiles.length > 0) {
      setPreview({
        type: 'video',
        files: selectedArtifact.videoFiles,
        artifact_id: selectedArtifact.artifact_id
      });
      setLoading(false);
      return;
    }

    // Handle file from file collection (clicked from Files panel)
    if (selectedArtifact.file) {
      const file = selectedArtifact.file;

      // Render file based on MIME type
      // Check CSV first before generic text check (CSV is text/csv so would be caught by text/* check)
      if (file.mime_type === 'text/csv' && file.content) {
        // Parse CSV and render as table
        Papa.parse(file.content, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {
            setPreview({
              type: 'table',
              columns: results.meta.fields,
              rows: results.data
            });
            setLoading(false);
          }
        });
        return;
      } else if (file.mime_type?.startsWith('image/')) {
        setPreview({
          type: 'image',
          files: [file],
          artifact_id: selectedArtifact.artifact_id
        });
      } else if (file.mime_type?.startsWith('audio/')) {
        setPreview({
          type: 'audio',
          files: [file],
          artifact_id: selectedArtifact.artifact_id
        });
      } else if (file.mime_type?.startsWith('video/')) {
        setPreview({
          type: 'video',
          files: [file],
          artifact_id: selectedArtifact.artifact_id
        });
      } else if (file.content) {
        // Text files - determine language from mime_type or extension
        let language = file.metadata?.language || 'text';
        if (file.mime_type === 'application/json') {
          language = 'json';
        } else if (file.mime_type === 'application/x-yaml') {
          language = 'yaml';
        } else if (file.path) {
          const ext = file.path.split('.').pop()?.toLowerCase();
          if (ext === 'py') language = 'python';
          else if (ext === 'js') language = 'javascript';
          else if (ext === 'ts') language = 'typescript';
          else if (ext === 'json') language = 'json';
          else if (ext === 'yaml' || ext === 'yml') language = 'yaml';
          else if (ext === 'md') language = 'markdown';
        }

        setPreview({
          type: 'text',
          content: file.content,
          language
        });
      } else {
        // No content available
        setPreview({
          type: 'text',
          content: '(Binary file - preview not available)',
          language: 'text'
        });
      }
      setLoading(false);
      return;
    }

    const fetchPreview = async () => {
      setLoading(true);
      try {
        const response = await apiClient.getArtifactPreview(selectedArtifact.artifact_id, offset, limit);

        const contentType = response.headers.get('content-type');

        // Handle CSV (text/csv)
        if (contentType?.includes('text/csv')) {
          const csvText = await response.text();
          const totalRows = parseInt(response.headers.get('x-total-rows') || '0', 10);
          const currentOffset = parseInt(response.headers.get('x-offset') || '0', 10);
          const currentLimit = parseInt(response.headers.get('x-limit') || '100', 10);
          const hasMore = response.headers.get('x-has-more') === 'true';

          // Parse CSV with Papa Parse
          Papa.parse(csvText, {
            header: true,
            skipEmptyLines: true,
            complete: (results) => {
              const previewData = {
                type: 'table',
                columns: results.meta.fields,
                rows: results.data,
                pagination: {
                  offset: currentOffset,
                  limit: currentLimit,
                  total_rows: totalRows,
                  has_more: hasMore
                }
              };
              setPreview(previewData);
              setLoading(false);
            },
            error: (error) => {
              console.error('CSV parse error:', error);
              setPreview(null);
              setLoading(false);
            }
          });
        }
        // Handle JSON (images, metadata, etc.)
        else if (contentType?.includes('application/json')) {
          const data = await response.json();
          setPreview(data);
          setLoading(false);
        }
        else {
          console.error('Unexpected content type:', contentType);
          setPreview(null);
          setLoading(false);
        }
      } catch (err) {
        console.error('Failed to fetch dataset preview:', err);
        setPreview(null);
        setLoading(false);
      }
    };

    fetchPreview();
  }, [selectedArtifact, offset, limit]);

  // Reset offset when artifact changes
  useEffect(() => {
    setOffset(0);
  }, [selectedArtifact]);

  if (!selectedArtifact) {
    return null;
  }

  if (loading) {
    return (
      <div className="artifact-tab-loading">
        Loading artifact preview...
      </div>
    );
  }

  const handlePrevPage = () => {
    setOffset(Math.max(0, offset - limit));
  };

  const handleNextPage = () => {
    setOffset(offset + limit);
  };

  return (
    <div className="artifact-tab">
      {/* Artifact header */}
      <div className="artifact-header">
        <div className="artifact-header-left">
          <h2>
            {selectedArtifact.folder
              ? `${selectedArtifact.folder}/`
              : selectedArtifact.file
              ? selectedArtifact.file.path
              : selectedArtifact.name}
          </h2>
          <div className="artifact-meta">
            {selectedArtifact.imageFiles || selectedArtifact.audioFiles || selectedArtifact.videoFiles ? (
              <span className="meta-item">
                <strong>Artifact:</strong> {selectedArtifact.name}
              </span>
            ) : selectedArtifact.file ? (
              <>
                <span className="meta-item">
                  <strong>Artifact:</strong> {selectedArtifact.name}
                </span>
                <span className="meta-item">
                  <strong>Size:</strong> {(selectedArtifact.file.size / 1024).toFixed(2)} KB
                </span>
                <span className="meta-item">
                  <strong>Type:</strong> {selectedArtifact.file.mime_type}
                </span>
              </>
            ) : (
              <>
                {selectedArtifact.hash && (
                  <span className="meta-item">
                    <strong>Hash:</strong> {selectedArtifact.hash}
                  </span>
                )}
                {selectedArtifact.size_bytes && (
                  <span className="meta-item">
                    <strong>Size:</strong> {(selectedArtifact.size_bytes / 1024).toFixed(2)} KB
                  </span>
                )}
                {selectedArtifact.metadata?.rows && (
                  <span className="meta-item">
                    <strong>Rows:</strong> {selectedArtifact.metadata.rows}
                  </span>
                )}
                {preview?.pagination?.total && (
                  <span className="meta-item">
                    <strong>Total:</strong> {preview.pagination.total}
                  </span>
                )}
              </>
            )}
            {/* Display flattened metadata fields from backend */}
            {selectedArtifact.metadata && Object.keys(selectedArtifact.metadata).length > 0 && (
              <>
                {Object.entries(selectedArtifact.metadata).map(([key, value]) => (
                  <span key={key} className="meta-item">
                    <strong>{key}:</strong> {value}
                  </span>
                ))}
              </>
            )}
          </div>
        </div>
        <div className="artifact-header-right">
          {selectedArtifact.imageFiles || selectedArtifact.audioFiles || selectedArtifact.videoFiles ? (
            // Download entire artifact for media folder
            <button
              className="download-btn"
              onClick={() => {
                const downloadUrl = `${API_BASE_URL}/api/artifact/${selectedArtifact.artifact_id}/download`;
                window.open(downloadUrl, '_blank');
              }}
            >
              Download All
            </button>
          ) : selectedArtifact.file ? (
            // Download individual file from file collection (content already in memory)
            <button
              className="download-btn"
              onClick={() => {
                const file = selectedArtifact.file;
                const blob = new Blob([file.content || ''], { type: file.mime_type || 'application/octet-stream' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = file.path.split('/').pop() || 'download';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
              }}
            >
              Download
            </button>
          ) : selectedArtifact.artifact_id ? (
            // Download entire artifact (single file or directory)
            <button
              className="download-btn"
              onClick={() => {
                const downloadUrl = `${API_BASE_URL}/api/artifact/${selectedArtifact.artifact_id}/download`;
                window.open(downloadUrl, '_blank');
              }}
            >
              Download
            </button>
          ) : null}
        </div>
      </div>

      {/* Preview content */}
      {preview?.type === 'table' && preview.columns && preview.rows ? (
        <>
          <div className="data-table-container">
            <table className="data-table">
              <thead>
                <tr>
                  {preview.columns.map(col => (
                    <th key={col}>{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.rows.map((row, idx) => (
                  <tr key={idx}>
                    {preview.columns.map(col => (
                      <td key={col}>{String(row[col] ?? '')}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {preview.pagination && (
            <div className="data-pagination">
              <button
                onClick={handlePrevPage}
                disabled={offset === 0}
                className="pagination-btn"
              >
                Previous
              </button>
              <span className="pagination-info">
                Rows {offset + 1} - {Math.min(offset + limit, preview.pagination.total_rows)} of {preview.pagination.total_rows}
              </span>
              <button
                onClick={handleNextPage}
                disabled={!preview.pagination.has_more}
                className="pagination-btn"
              >
                Next
              </button>
            </div>
          )}
        </>
      ) : preview?.type === 'image' && preview.files?.length > 0 ? (
        <>
          <div className="image-grid">
            {preview.files.map((file, idx) => {
              // Handle both file objects (from file collection) and string paths (from API)
              const isFileObject = typeof file === 'object' && file !== null;

              let imageSrc, imageLabel;
              if (isFileObject) {
                // File object from file collection
                // Use API endpoint to fetch the file from artifact storage
                const filePath = file.path || file.displayName;
                imageSrc = `${API_BASE_URL}/api/artifact/${preview.artifact_id}/files/${encodeURIComponent(filePath)}`;
                imageLabel = String(file.displayName || file.path || 'image');
              } else {
                // String path from API
                imageSrc = `${API_BASE_URL}${preview.base_url}/${file}`;
                imageLabel = String(file);
              }

              return (
                <div key={idx} className="image-grid-item">
                  <img
                    src={imageSrc}
                    alt={imageLabel}
                    loading="lazy"
                  />
                  <div className="image-filename">{imageLabel}</div>
                </div>
              );
            })}
          </div>

          {/* Pagination for images */}
          {preview.pagination && (
            <div className="data-pagination">
              <button
                onClick={handlePrevPage}
                disabled={offset === 0}
                className="pagination-btn"
              >
                Previous
              </button>
              <span className="pagination-info">
                Images {offset + 1} - {Math.min(offset + limit, preview.pagination.total)} of {preview.pagination.total}
              </span>
              <button
                onClick={handleNextPage}
                disabled={!preview.pagination.has_more}
                className="pagination-btn"
              >
                Next
              </button>
            </div>
          )}
        </>
      ) : preview?.type === 'audio' && preview.files?.length > 0 ? (
        <div className="media-grid">
          {preview.files.map((file, idx) => {
            const filePath = file.path || file.displayName;
            const audioSrc = `${API_BASE_URL}/api/artifact/${preview.artifact_id}/files/${encodeURIComponent(filePath)}`;
            return (
              <div key={idx} className="media-grid-item">
                <audio controls src={audioSrc} />
                <div className="media-filename">{file.displayName || file.path}</div>
              </div>
            );
          })}
        </div>
      ) : preview?.type === 'video' && preview.files?.length > 0 ? (
        <div className="media-grid">
          {preview.files.map((file, idx) => {
            const filePath = file.path || file.displayName;
            const videoSrc = `${API_BASE_URL}/api/artifact/${preview.artifact_id}/files/${encodeURIComponent(filePath)}`;
            return (
              <div key={idx} className="media-grid-item">
                <video controls src={videoSrc} />
                <div className="media-filename">{file.displayName || file.path}</div>
              </div>
            );
          })}
        </div>
      ) : preview?.type === 'text' && preview.content ? (
        <div className="text-preview-container">
          <SyntaxHighlighter
            language={preview.language || 'text'}
            style={prism}
            showLineNumbers={true}
            wrapLines={true}
            customStyle={{
              margin: 0,
              borderRadius: 0,
              fontSize: '13px',
              height: '100%',
              backgroundColor: '#ffffff'
            }}
          >
            {preview.content}
          </SyntaxHighlighter>
        </div>
      ) : preview?.download_url ? (
        <div className="artifact-tab-empty">
          <div className="empty-state">
            <div className="empty-title">No preview available</div>
            <div className="empty-message">
              {preview?.message}
            </div>
            <a
              href={`${API_BASE_URL}${preview.download_url}`}
              className="download-link"
              download
            >
              Download to view
            </a>
          </div>
        </div>
      ) : null}
    </div>
  );
};
