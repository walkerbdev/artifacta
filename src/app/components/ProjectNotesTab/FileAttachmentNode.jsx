import React from 'react';
import { NodeViewWrapper } from '@tiptap/react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { prism } from 'react-syntax-highlighter/dist/esm/styles/prism';

/**
 * File Attachment Node component for TipTap editor
 *
 * Custom TipTap node that renders file attachments with appropriate previews
 * directly inline in the editor. Used by FileAttachmentExtension.
 *
 * Supported file types:
 * - Code/text files: Syntax-highlighted preview (Python, JS, JSON, etc.)
 * - PDFs: Embedded iframe viewer
 * - Videos: HTML5 video player (MP4, WEBM)
 * - Audio: HTML5 audio player (MP3, WAV)
 * - Other files: Download card with file info
 *
 * Features:
 * - Type-specific rendering based on MIME type
 * - Syntax highlighting for code files (Prism)
 * - Inline preview for media files
 * - Download button for all files
 * - File size display
 * - Responsive layouts
 *
 * @param {object} props - Component props
 * @param {object} props.node - TipTap node object with attrs:
 *   - url: string - File URL for download/preview
 *   - fileName: string - Display name
 *   - fileSize: number - Size in KB
 *   - fileType: string - MIME type
 *   - textContent: string (optional) - For code preview
 *   - language: string (optional) - Syntax highlighting language
 * @returns {React.ReactElement} Rendered file attachment with preview
 */
export const FileAttachmentComponent = ({ node }) => {
  const { url, fileName, fileSize, fileType, textContent, language } = node.attrs;

  // Code/text files with syntax highlighting
  if (textContent && language) {
    return (
      <NodeViewWrapper className="inline-file-preview">
        <div style={{ maxHeight: '400px', overflow: 'auto', border: '1px solid #e5e7eb', borderRadius: '6px' }}>
          <SyntaxHighlighter
            language={language}
            style={prism}
            showLineNumbers
            wrapLines
            customStyle={{
              margin: 0,
              borderRadius: '6px',
              fontSize: '13px',
              backgroundColor: '#ffffff',
            }}
          >
            {textContent}
          </SyntaxHighlighter>
        </div>
        <div className="file-caption">📄 {fileName} ({fileSize} KB)</div>
      </NodeViewWrapper>
    );
  }

  // PDF preview
  if (fileType === 'application/pdf') {
    return (
      <NodeViewWrapper className="inline-file-preview">
        <iframe
          src={url}
          width="100%"
          height="600"
          style={{ border: '1px solid #e5e7eb', borderRadius: '6px' }}
          title={fileName}
        />
        <div className="file-caption">📄 {fileName} ({fileSize} KB)</div>
      </NodeViewWrapper>
    );
  }

  // Video preview
  if (fileType?.startsWith('video/')) {
    return (
      <NodeViewWrapper className="inline-file-preview">
        <video controls style={{ width: '100%', borderRadius: '6px', maxHeight: '500px' }}>
          <source src={url} type={fileType} />
        </video>
        <div className="file-caption">🎥 {fileName} ({fileSize} KB)</div>
      </NodeViewWrapper>
    );
  }

  // Audio preview
  if (fileType?.startsWith('audio/')) {
    return (
      <NodeViewWrapper className="inline-file-preview">
        <audio controls style={{ width: '100%' }}>
          <source src={url} type={fileType} />
        </audio>
        <div className="file-caption">🎵 {fileName} ({fileSize} KB)</div>
      </NodeViewWrapper>
    );
  }

  // Default file card with download button
  return (
    <NodeViewWrapper>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '12px 16px',
          margin: '8px 0',
          background: '#f9fafb',
          border: '1px solid #e5e7eb',
          borderRadius: '8px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flex: 1, minWidth: 0 }}>
          <span style={{ fontSize: '20px' }}>📎</span>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div
              style={{
                color: '#111827',
                fontWeight: 500,
                fontSize: '14px',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              {fileName}
            </div>
            <div style={{ color: '#6b7280', fontSize: '12px', marginTop: '2px' }}>
              {fileSize} KB
            </div>
          </div>
        </div>
        {url && (
          <a
            href={url}
            download={fileName}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 12px',
              background: '#fff',
              color: '#374151',
              borderRadius: '6px',
              textDecoration: 'none',
              fontSize: '13px',
              fontWeight: 500,
              border: '1px solid #d1d5db',
              transition: 'all 0.2s',
              whiteSpace: 'nowrap',
            }}
            onMouseEnter={(e) => {
              e.target.style.background = '#f3f4f6';
              e.target.style.borderColor = '#9ca3af';
            }}
            onMouseLeave={(e) => {
              e.target.style.background = '#fff';
              e.target.style.borderColor = '#d1d5db';
            }}
          >
            <span>⬇</span>
            <span>Download</span>
          </a>
        )}
      </div>
    </NodeViewWrapper>
  );
};
