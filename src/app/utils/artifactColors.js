/**
 * Artifact color scheme based on file extension
 * Returns a color for visual distinction in lineage graphs and artifact panels
 */

// Extension to color mapping
const EXTENSION_COLORS = {
  // Code files
  py: '#3776ab',      // Python blue
  js: '#f7df1e',      // JavaScript yellow
  jsx: '#61dafb',     // React cyan
  ts: '#3178c6',      // TypeScript blue
  tsx: '#3178c6',     // TypeScript blue
  java: '#007396',    // Java blue
  cpp: '#00599c',     // C++ blue
  c: '#a8b9cc',       // C gray-blue
  rs: '#ce422b',      // Rust orange
  go: '#00add8',      // Go cyan
  rb: '#cc342d',      // Ruby red
  php: '#777bb4',     // PHP purple
  swift: '#fa7343',   // Swift orange
  kt: '#7f52ff',      // Kotlin purple

  // Data files
  json: '#5382a1',    // JSON blue
  yaml: '#cb171e',    // YAML red
  yml: '#cb171e',     // YAML red
  xml: '#e34c26',     // XML orange
  csv: '#17a2b8',     // CSV teal
  tsv: '#17a2b8',     // TSV teal
  parquet: '#0c7bb3', // Parquet blue

  // Model files
  pt: '#ee4c2c',      // PyTorch red
  pth: '#ee4c2c',     // PyTorch red
  ckpt: '#ff6f00',    // Checkpoint orange
  h5: '#d00000',      // HDF5/Keras red
  pb: '#ff6f00',      // TensorFlow protobuf orange
  onnx: '#005CED',    // ONNX blue
  pkl: '#306998',     // Pickle blue
  joblib: '#306998',  // Joblib blue

  // Config files
  cfg: '#6c757d',     // Config gray
  conf: '#6c757d',    // Config gray
  ini: '#6c757d',     // INI gray
  toml: '#9c4121',    // TOML brown
  env: '#4caf50',     // ENV green

  // Documents
  md: '#083fa1',      // Markdown blue
  txt: '#495057',     // Text gray
  pdf: '#f40f02',     // PDF red

  // Images
  png: '#ffc107',     // Image yellow
  jpg: '#ffc107',     // Image yellow
  jpeg: '#ffc107',    // Image yellow
  gif: '#ffc107',     // Image yellow
  svg: '#ffb13b',     // SVG orange-yellow

  // Default fallback
  default: '#6c757d' // Gray
};

/**
 * Get color for an artifact based on its name/path
 * @param {string} name - Artifact name or file path
 * @returns {string} Hex color code
 */
export function getArtifactColor(name) {
  if (!name) return EXTENSION_COLORS.default;

  // Extract extension from name (handle both files and directories)
  const parts = name.split('.');
  if (parts.length < 2) {
    // No extension - could be a directory or file without extension
    return EXTENSION_COLORS.default;
  }

  const ext = parts[parts.length - 1].toLowerCase();
  return EXTENSION_COLORS[ext] || EXTENSION_COLORS.default;
}

/**
 * Get all available extension colors (for reference/debugging)
 * @returns {Object} Extension to color mapping
 */
export function getAllExtensionColors() {
  return { ...EXTENSION_COLORS };
}
