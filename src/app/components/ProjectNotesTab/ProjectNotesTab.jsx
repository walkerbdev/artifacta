import React, { useState, useEffect } from 'react';
import { HiCheck, HiTrash } from 'react-icons/hi';
import {
  HiBold, HiListBullet, HiCodeBracket,
} from 'react-icons/hi2';
import {
  MdFormatItalic, MdFormatListNumbered, MdFunctions, MdCode,
  MdFormatQuote, MdHorizontalRule, MdLink,
  MdUndo, MdRedo, MdStrikethroughS,
} from 'react-icons/md';
import {
  BsTable, BsPlus, BsTrash, BsPaperclip,
} from 'react-icons/bs';
import { useEditor, EditorContent } from '@tiptap/react';
import StarterKit from '@tiptap/starter-kit';
import Image from '@tiptap/extension-image';
import Link from '@tiptap/extension-link';
import TaskList from '@tiptap/extension-task-list';
import TaskItem from '@tiptap/extension-task-item';
import { Table } from '@tiptap/extension-table';
import { TableRow } from '@tiptap/extension-table-row';
import { TableHeader } from '@tiptap/extension-table-header';
import { TableCell } from '@tiptap/extension-table-cell';
import { MathExtension } from '@aarkue/tiptap-math-extension';
import 'katex/dist/katex.min.css';
import { apiClient } from '@/core/api/ApiClient';
import { FileAttachment } from './FileAttachmentExtension';
import './ProjectNotesTab.scss';

const MenuBar = ({ editor, onFileUpload }) => {
  const [showTableMenu, setShowTableMenu] = React.useState(false);
  const [showHeadingMenu, setShowHeadingMenu] = React.useState(false);
  const tableMenuRef = React.useRef(null);
  const headingMenuRef = React.useRef(null);
  const fileInputRef = React.useRef(null);

  // Close dropdowns when clicking outside
  React.useEffect(() => {
    const handleClickOutside = (event) => {
      if (tableMenuRef.current && !tableMenuRef.current.contains(event.target)) {
        setShowTableMenu(false);
      }
      if (headingMenuRef.current && !headingMenuRef.current.contains(event.target)) {
        setShowHeadingMenu(false);
      }
    };

    if (showTableMenu || showHeadingMenu) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showTableMenu, showHeadingMenu]);

  if (!editor) {
    return null;
  }

  const ToolbarButton = ({ onClick, isActive, children, title }) => (
    <button
      onClick={onClick}
      className={`toolbar-btn ${isActive ? 'active' : ''}`}
      title={title}
      type="button"
    >
      {children}
    </button>
  );

  const getHeadingLabel = () => {
    if (editor.isActive('heading', { level: 1 })) return 'H1';
    if (editor.isActive('heading', { level: 2 })) return 'H2';
    if (editor.isActive('heading', { level: 3 })) return 'H3';
    if (editor.isActive('heading', { level: 4 })) return 'H4';
    if (editor.isActive('heading', { level: 5 })) return 'H5';
    if (editor.isActive('heading', { level: 6 })) return 'H6';
    return 'H';
  };

  return (
    <div className="editor-toolbar">
      <div className="toolbar-group heading-dropdown-wrapper" ref={headingMenuRef}>
        <button
          onClick={() => setShowHeadingMenu(!showHeadingMenu)}
          className="toolbar-btn"
          title="Text style"
          type="button"
        >
          {getHeadingLabel()}
        </button>
        {showHeadingMenu && (
          <div className="heading-dropdown-menu">
            <button
              onClick={() => {
                editor.chain().focus().setParagraph().run();
                setShowHeadingMenu(false);
              }}
              className={editor.isActive('paragraph') ? 'active' : ''}
              type="button"
            >
              Normal Text
            </button>
            <button
              onClick={() => {
                editor.chain().focus().toggleHeading({ level: 1 }).run();
                setShowHeadingMenu(false);
              }}
              className={editor.isActive('heading', { level: 1 }) ? 'active' : ''}
              type="button"
            >
              Heading 1
            </button>
            <button
              onClick={() => {
                editor.chain().focus().toggleHeading({ level: 2 }).run();
                setShowHeadingMenu(false);
              }}
              className={editor.isActive('heading', { level: 2 }) ? 'active' : ''}
              type="button"
            >
              Heading 2
            </button>
            <button
              onClick={() => {
                editor.chain().focus().toggleHeading({ level: 3 }).run();
                setShowHeadingMenu(false);
              }}
              className={editor.isActive('heading', { level: 3 }) ? 'active' : ''}
              type="button"
            >
              Heading 3
            </button>
            <button
              onClick={() => {
                editor.chain().focus().toggleHeading({ level: 4 }).run();
                setShowHeadingMenu(false);
              }}
              className={editor.isActive('heading', { level: 4 }) ? 'active' : ''}
              type="button"
            >
              Heading 4
            </button>
            <button
              onClick={() => {
                editor.chain().focus().toggleHeading({ level: 5 }).run();
                setShowHeadingMenu(false);
              }}
              className={editor.isActive('heading', { level: 5 }) ? 'active' : ''}
              type="button"
            >
              Heading 5
            </button>
            <button
              onClick={() => {
                editor.chain().focus().toggleHeading({ level: 6 }).run();
                setShowHeadingMenu(false);
              }}
              className={editor.isActive('heading', { level: 6 }) ? 'active' : ''}
              type="button"
            >
              Heading 6
            </button>
          </div>
        )}
      </div>

      <div className="toolbar-separator" />

      <div className="toolbar-group">
        <ToolbarButton
          onClick={() => editor.chain().focus().toggleBold().run()}
          isActive={editor.isActive('bold')}
          title="Bold (Cmd+B)"
        >
          <HiBold />
        </ToolbarButton>
        <ToolbarButton
          onClick={() => editor.chain().focus().toggleItalic().run()}
          isActive={editor.isActive('italic')}
          title="Italic (Cmd+I)"
        >
          <MdFormatItalic />
        </ToolbarButton>
        <ToolbarButton
          onClick={() => editor.chain().focus().toggleStrike().run()}
          isActive={editor.isActive('strike')}
          title="Strikethrough"
        >
          <MdStrikethroughS />
        </ToolbarButton>
        <ToolbarButton
          onClick={() => editor.chain().focus().toggleCode().run()}
          isActive={editor.isActive('code')}
          title="Inline Code"
        >
          <MdCode />
        </ToolbarButton>
      </div>

      <div className="toolbar-separator" />

      <div className="toolbar-group">
        <ToolbarButton
          onClick={() => editor.chain().focus().toggleBulletList().run()}
          isActive={editor.isActive('bulletList')}
          title="Bullet List"
        >
          <HiListBullet />
        </ToolbarButton>
        <ToolbarButton
          onClick={() => editor.chain().focus().toggleOrderedList().run()}
          isActive={editor.isActive('orderedList')}
          title="Numbered List"
        >
          <MdFormatListNumbered />
        </ToolbarButton>
        <ToolbarButton
          onClick={() => editor.chain().focus().toggleBlockquote().run()}
          isActive={editor.isActive('blockquote')}
          title="Blockquote"
        >
          <MdFormatQuote />
        </ToolbarButton>
      </div>

      <div className="toolbar-separator" />

      <div className="toolbar-group">
        <ToolbarButton
          onClick={() => {
            const url = window.prompt('Enter URL:');
            if (url) {
              editor.chain().focus().setLink({ href: url }).run();
            }
          }}
          isActive={editor.isActive('link')}
          title="Insert Link"
        >
          <MdLink />
        </ToolbarButton>
        <ToolbarButton
          onClick={() => editor.chain().focus().setHorizontalRule().run()}
          isActive={false}
          title="Horizontal Line"
        >
          <MdHorizontalRule />
        </ToolbarButton>
      </div>

      <div className="toolbar-separator" />

      <div className="toolbar-group">
        <ToolbarButton
          onClick={() => editor.chain().focus().toggleCodeBlock().run()}
          isActive={editor.isActive('codeBlock')}
          title="Code Block"
        >
          <HiCodeBracket />
        </ToolbarButton>
        <ToolbarButton
          onClick={() => {
            // Insert inline math node
            editor.chain().focus().insertContent({
              type: 'inlineMath',
              attrs: {
                latex: 'x = y',
                display: 'no',
                evaluate: 'no'
              }
            }).run();
          }}
          isActive={editor.isActive('inlineMath')}
          title="Insert Math Equation (or type $...$)"
        >
          <MdFunctions />
        </ToolbarButton>
      </div>

      <div className="toolbar-separator" />

      <div className="toolbar-group table-dropdown-wrapper" ref={tableMenuRef}>
        <button
          onClick={() => setShowTableMenu(!showTableMenu)}
          className="toolbar-btn"
          title="Table options"
          type="button"
        >
          <BsTable />
        </button>
        {showTableMenu && (
          <div className="table-dropdown-menu">
            <button
              onClick={() => {
                editor.chain().focus().insertTable({ rows: 3, cols: 3, withHeaderRow: true }).run();
                setShowTableMenu(false);
              }}
              type="button"
            >
              <BsTable /> Insert Table
            </button>
            <button
              onClick={() => {
                editor.chain().focus().addRowAfter().run();
                setShowTableMenu(false);
              }}
              type="button"
            >
              <BsPlus /> Add Row
            </button>
            <button
              onClick={() => {
                editor.chain().focus().addColumnAfter().run();
                setShowTableMenu(false);
              }}
              type="button"
            >
              <BsPlus /> Add Column
            </button>
            <button
              onClick={() => {
                editor.chain().focus().deleteRow().run();
                setShowTableMenu(false);
              }}
              type="button"
            >
              <BsTrash /> Delete Row
            </button>
            <button
              onClick={() => {
                editor.chain().focus().deleteColumn().run();
                setShowTableMenu(false);
              }}
              type="button"
            >
              <BsTrash /> Delete Column
            </button>
            <button
              onClick={() => {
                editor.chain().focus().deleteTable().run();
                setShowTableMenu(false);
              }}
              type="button"
            >
              <BsTrash /> Delete Table
            </button>
          </div>
        )}
      </div>

      <div className="toolbar-separator" />

      <div className="toolbar-group">
        <input
          ref={fileInputRef}
          type="file"
          style={{ display: 'none' }}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file && onFileUpload) {
              onFileUpload(file);
            }
            e.target.value = ''; // Reset input
          }}
        />
        <ToolbarButton
          onClick={() => fileInputRef.current?.click()}
          isActive={false}
          title="Upload File or Image"
        >
          <BsPaperclip />
        </ToolbarButton>
      </div>

      <div className="toolbar-separator" />

      <div className="toolbar-group">
        <ToolbarButton
          onClick={() => editor.chain().focus().undo().run()}
          isActive={false}
          title="Undo (Cmd+Z)"
        >
          <MdUndo />
        </ToolbarButton>
        <ToolbarButton
          onClick={() => editor.chain().focus().redo().run()}
          isActive={false}
          title="Redo (Cmd+Shift+Z)"
        >
          <MdRedo />
        </ToolbarButton>
      </div>
    </div>
  );
};

export const ProjectNotesTab = ({
  projectId,
  availableRuns: _availableRuns = [],
  externalNoteId = null,  // Note ID to load from sidebar
  externalProjectId = null, // Project ID to load from sidebar
  isCreatingNew = false  // Flag to start creating new note
}) => {
  const [selectedNote, setSelectedNote] = useState(null);
  const [isCreating, setIsCreating] = useState(false);

  // Form state
  const [title, setTitle] = useState('');

  // TipTap editor
  const editor = useEditor({
    extensions: [
      StarterKit.configure({
        // Disable the default link extension from StarterKit
        link: false,
      }),
      Link.configure({
        openOnClick: false,
      }),
      Image,
      TaskList,
      TaskItem.configure({
        nested: true,
      }),
      Table.configure({
        resizable: true,
      }),
      TableRow,
      TableHeader,
      TableCell,
      MathExtension.configure({
        delimiters: 'dollar', // Enable $...$ for inline and $$...$$ for display
        evaluation: false,
      }),
      FileAttachment,
    ],
    content: '',
    editorProps: {
      attributes: {
        class: 'prose prose-sm max-w-none focus:outline-none min-h-[200px] p-4',
      },
    },
  });

  // Handle external note selection from sidebar
  useEffect(() => {
    if (externalNoteId && externalProjectId) {
      loadNote(externalProjectId, externalNoteId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [externalNoteId, externalProjectId]);

  // Handle external new note request from sidebar
  useEffect(() => {
    if (isCreatingNew && externalProjectId) {
      setIsCreating(true);
      setSelectedNote(null);
      resetForm();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isCreatingNew, externalProjectId]);


  const loadNote = async (activeProjectId, noteId) => {
    try {
      const note = await apiClient.getProjectNote(activeProjectId, noteId);
      setSelectedNote(note);
      setTitle(note.title);
      if (editor) {
        // Set the note content
        editor.commands.setContent(note.content || '');

        // Append attachments to the editor if they exist
        // Use setTimeout to avoid React flushSync warning by deferring until after render
        if (note.attachments && note.attachments.length > 0) {
          setTimeout(async () => {
            const API_BASE_URL = import.meta.env.VITE_API_URL;

            // Process each attachment
            for (const attachment of note.attachments) {
            const downloadUrl = `${API_BASE_URL}/api/attachments/${attachment.id}/download`;
            const fileSize = (attachment.filesize / 1024).toFixed(1);

            // Check if this is a text/code file that needs content fetching
            const isImage = attachment.mime_type.startsWith('image/');
            const textExtensions = ['.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.rb', '.go', '.rs', '.php', '.swift', '.kt', '.sql', '.sh', '.bash', '.json', '.xml', '.yaml', '.yml', '.md', '.html', '.css', '.scss', '.txt', '.log'];
            const isTextFile = textExtensions.some(ext => attachment.real_name.toLowerCase().endsWith(ext));

            if (isImage) {
              // Images: Insert as image node at the end
              editor
                .chain()
                .focus('end')
                .insertContent({
                  type: 'image',
                  attrs: {
                    src: downloadUrl,
                    alt: attachment.real_name,
                  },
                })
                .run();
            } else if (isTextFile) {
              // Text files: Fetch content and inject with syntax highlighting
              try {
                const response = await fetch(downloadUrl);
                const textContent = await response.text();
                const language = getLanguageFromFilename(attachment.real_name);

                editor
                  .chain()
                  .focus('end')
                  .insertContent({
                    type: 'fileAttachment',
                    attrs: {
                      textContent,
                      language,
                      fileName: attachment.real_name,
                      fileSize,
                      fileType: attachment.mime_type,
                    },
                  })
                  .run();
              } catch (err) {
                console.error(`Failed to fetch text content for ${attachment.real_name}:`, err);
              }
            } else {
              // PDFs, videos, audio: Use download URL directly
              editor
                .chain()
                .focus('end')
                .insertContent({
                  type: 'fileAttachment',
                  attrs: {
                    url: downloadUrl,
                    fileName: attachment.real_name,
                    fileSize: fileSize,
                    fileType: attachment.mime_type,
                  },
                })
                .run();
            }
          } // End of for loop
          }, 0); // End of setTimeout
        }
      }
      setIsCreating(false);
    } catch (error) {
      console.error('Failed to load note:', error);
    }
  };

  const createNote = async () => {
    try {
      const activeProjectId = externalProjectId || projectId;
      const content = editor ? editor.getHTML() : '';

      const newNote = await apiClient.createProjectNote(activeProjectId, {
        title,
        content
      });
      setIsCreating(false);
      loadNote(activeProjectId, newNote.id);
      resetForm();

      window.dispatchEvent(new Event('refreshProjectNotes'));
    } catch (error) {
      console.error('Failed to create note:', error);
    }
  };

  const updateNote = async () => {
    try {
      const activeProjectId = externalProjectId || projectId;
      const content = editor ? editor.getHTML() : '';

      await apiClient.updateProjectNote(activeProjectId, selectedNote.id, {
        title,
        content
      });
      loadNote(activeProjectId, selectedNote.id);
    } catch (error) {
      console.error('Failed to update note:', error);
    }
  };

  const deleteNote = async (noteId) => {
    // eslint-disable-next-line no-undef
    if (!confirm('Delete this note? This cannot be undone.')) return;

    try {
      const activeProjectId = externalProjectId || projectId;
      await apiClient.deleteProjectNote(activeProjectId, noteId);
      if (selectedNote?.id === noteId) {
        setSelectedNote(null);
        setIsCreating(false);
        resetForm();
      }

      // Trigger refresh of projects panel
      window.dispatchEvent(new Event('refreshProjectNotes'));
    } catch (error) {
      console.error('Failed to delete note:', error);
    }
  };

  const resetForm = () => {
    setTitle('');
    if (editor) {
      editor.commands.setContent('');
    }
  };

  // Helper: Detect language from file extension
  const getLanguageFromFilename = (filename) => {
    const ext = filename.split('.').pop()?.toLowerCase();
    const langMap = {
      js: 'javascript', jsx: 'javascript', ts: 'typescript', tsx: 'typescript',
      py: 'python', java: 'java', cpp: 'cpp', c: 'c', cs: 'csharp',
      rb: 'ruby', go: 'go', rs: 'rust', php: 'php', swift: 'swift',
      kt: 'kotlin', sql: 'sql', sh: 'bash', bash: 'bash',
      json: 'json', xml: 'xml', yaml: 'yaml', yml: 'yaml',
      md: 'markdown', html: 'html', css: 'css', scss: 'scss',
      txt: 'text', log: 'text',
    };
    return langMap[ext] || 'text';
  };

  // Helper: Check if file is a text/code file
  const isTextFile = (file) => {
    const textTypes = ['text/', 'application/json', 'application/xml', 'application/javascript'];
    const textExtensions = ['.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.rb', '.go', '.rs', '.php', '.swift', '.kt', '.sql', '.sh', '.bash', '.json', '.xml', '.yaml', '.yml', '.md', '.html', '.css', '.scss', '.txt', '.log'];

    return textTypes.some(type => file.type.startsWith(type)) ||
           textExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
  };

  const handleFileUpload = async (file) => {
    try {
      const isImage = file.type.startsWith('image/');
      const isText = isTextFile(file);
      const canEmbed = isImage ||
                      file.type === 'application/pdf' ||
                      file.type.startsWith('video/') ||
                      file.type.startsWith('audio/') ||
                      isText;

      if (canEmbed) {
        const fileSize = (file.size / 1024).toFixed(1);

        if (isText) {
          // Text/code files: Read as text for syntax highlighting
          const textReader = new FileReader();
          textReader.onload = () => {
            const textContent = textReader.result;
            const language = getLanguageFromFilename(file.name);

            editor
              .chain()
              .focus()
              .insertContent({
                type: 'fileAttachment',
                attrs: {
                  textContent,
                  language,
                  fileName: file.name,
                  fileSize,
                  fileType: file.type,
                },
              })
              .run();
          };
          textReader.readAsText(file);
        } else {
          // Binary files (images, PDFs, videos, audio): Read as data URL
          const reader = new FileReader();
          reader.onload = () => {
            const dataUrl = reader.result;

            if (isImage) {
              // Images: Use built-in TipTap image node
              editor.chain().focus().setImage({ src: dataUrl }).run();
            } else {
              // PDFs, videos, audio: Use custom file attachment node
              editor
                .chain()
                .focus()
                .insertContent({
                  type: 'fileAttachment',
                  attrs: {
                    url: dataUrl,
                    fileName: file.name,
                    fileSize,
                    fileType: file.type,
                  },
                })
                .run();
            }
          };
          reader.readAsDataURL(file);
        }
      } else {
        // For other files: Need a saved note to upload attachments
        if (!selectedNote) {
          window.alert('Please save the note first before uploading file attachments');
          return;
        }
        // For other files: Upload as attachment
        const formData = new FormData();
        formData.append('file', file);

        const API_BASE_URL = import.meta.env.VITE_API_URL;
        const response = await fetch(`${API_BASE_URL}/api/notes/${selectedNote.id}/attachments`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) throw new Error('Upload failed');

        // Get the uploaded attachment data
        const uploadedFile = await response.json();
        const downloadUrl = `${API_BASE_URL}/api/attachments/${uploadedFile.id}/download`;
        const fileSize = (file.size / 1024).toFixed(1);

        // Insert file attachment using custom TipTap node
        editor
          .chain()
          .focus()
          .insertContent({
            type: 'fileAttachment',
            attrs: {
              url: downloadUrl,
              fileName: file.name,
              fileSize: fileSize,
              fileType: file.type,
            },
          })
          .run();
      }
    } catch (error) {
      console.error('File upload failed:', error);
      window.alert('Failed to upload file');
    }
  };

  if (!projectId) {
    return (
      <div className="project-notes-empty">
        <p>Select a project to view notes</p>
      </div>
    );
  }

  return (
    <div className="project-notes-tab">
      {/* Note editor/viewer - full width */}
      <div className="notes-main-fullwidth">
        {(isCreating || selectedNote) ? (
          <>
            {/* Header with title and actions */}
            <div className="note-header">
              <input
                type="text"
                className="note-title-input"
                placeholder="Untitled note"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
              />

              <div className="note-actions">
                <button
                  onClick={isCreating ? createNote : updateNote}
                  className="btn-save"
                  disabled={!title.trim()}
                  title={isCreating ? 'Create note' : 'Save changes'}
                >
                  <HiCheck />
                </button>
                {!isCreating && (
                  <button
                    onClick={() => deleteNote(selectedNote.id)}
                    className="btn-delete-note"
                    title="Delete note"
                  >
                    <HiTrash />
                  </button>
                )}
              </div>
            </div>

            {/* TipTap editor */}
            <div className="tiptap-container">
              <MenuBar editor={editor} onFileUpload={handleFileUpload} />
              <EditorContent editor={editor} />
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
};
