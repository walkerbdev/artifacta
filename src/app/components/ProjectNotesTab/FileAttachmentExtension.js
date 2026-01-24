import { Node } from '@tiptap/core';
import { ReactNodeViewRenderer } from '@tiptap/react';
import { FileAttachmentComponent } from './FileAttachmentNode';

export const FileAttachment = Node.create({
  name: 'fileAttachment',

  group: 'block',

  atom: true,

  /**
   * Defines the attributes for the file attachment node.
   * @returns {object} The attributes object containing url, fileName, fileSize, fileType, textContent, and language.
   */
  addAttributes() {
    return {
      url: { default: null },
      fileName: { default: null },
      fileSize: { default: null },
      fileType: { default: null },
      textContent: { default: null },
      language: { default: null },
    };
  },

  /**
   * Parses HTML to recognize file attachment elements.
   * @returns {Array<object>} Array of parsing rules for HTML elements.
   */
  parseHTML() {
    return [{ tag: 'div[data-file-attachment]' }];
  },

  /**
   * Renders the file attachment node as HTML.
   * @param {object} options - The rendering options.
   * @param {object} options.HTMLAttributes - The HTML attributes to apply to the element.
   * @returns {Array} Array containing the tag name and attributes for rendering.
   */
  renderHTML({ HTMLAttributes }) {
    return ['div', { 'data-file-attachment': '', ...HTMLAttributes }];
  },

  /**
   * Adds a custom React node view for the file attachment.
   * @returns {object} The React node view renderer for the file attachment component.
   */
  addNodeView() {
    return ReactNodeViewRenderer(FileAttachmentComponent);
  },
});
