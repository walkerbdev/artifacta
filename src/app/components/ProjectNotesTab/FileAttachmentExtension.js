import { Node } from '@tiptap/core';
import { ReactNodeViewRenderer } from '@tiptap/react';
import { FileAttachmentComponent } from './FileAttachmentNode';

export const FileAttachment = Node.create({
  name: 'fileAttachment',

  group: 'block',

  atom: true,

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

  parseHTML() {
    return [{ tag: 'div[data-file-attachment]' }];
  },

  renderHTML({ HTMLAttributes }) {
    return ['div', { 'data-file-attachment': '', ...HTMLAttributes }];
  },

  addNodeView() {
    return ReactNodeViewRenderer(FileAttachmentComponent);
  },
});
