/**
 * Shared styling constants for sidebar collapsible components
 * (ProjectsPanel, RunTree, ArtifactsPanel)
 */

export const SIDEBAR_STYLES = {
  // Font family for sidebar text (file names, run names, project names)
  fontFamily: "inherit",

  // Font sizes
  fontSize: {
    header: '0.875rem',    // Collapsible headers (projects, runs, artifacts)
    item: '0.875rem',      // Individual items (run names, file names)
    button: '0.875rem',    // Action buttons
    metadata: '13px',      // Secondary info (file sizes, run counts)
  },

  // Font weights
  fontWeight: {
    header: 500,
    item: 400,
    button: 500,
  },

  // Spacing
  spacing: {
    padding: '6px 8px',
    gap: '6px',
    indent: '16px',
  },
};
