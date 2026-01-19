/**
 * Chart & Visualization Constants
 */

/**
 * Standard padding for all chart components
 * Ensures consistent spacing and alignment across visualizations
 */
export const CHART_PADDING = {
  top: 20,
  right: 20,
  bottom: 50,
  left: 80
};

/**
 * Standard color palette for charts and visualizations
 * Ensures consistent colors across all plot components
 */
const CHART_COLORS = [
  '#4A90E2', // Blue
  '#E94B3C', // Red
  '#6BCF7F', // Green
  '#F5A623', // Orange
  '#9B59B6', // Purple
  '#1ABC9C', // Turquoise
  '#E67E22', // Dark Orange
  '#34495E', // Dark Blue-Gray
  '#E74C3C', // Crimson
  '#3498DB', // Light Blue
  '#2ECC71', // Emerald
  '#F39C12'  // Yellow-Orange
];

/**
 * Get color for dataset by index (wraps around if more datasets than colors)
 * @param {number} index - Dataset index
 * @returns {string} Hex color code
 */
export function getChartColor(index) {
  return CHART_COLORS[index % CHART_COLORS.length];
}
