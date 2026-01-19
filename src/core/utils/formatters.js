/**
 * Format utility functions for displaying values across the application
 * Centralizes formatting logic to ensure consistency
 */

/**
 * Format number for Y-axis labels with smart units
 * Handles bytes specially, and uses K/M/B/T suffixes for large numbers
 *
 * @param {number} value - The value to format
 * @param {string} title - Optional title to determine if value represents bytes
 * @returns {string} Formatted string
 */
export function formatYAxisValue(value, title = '') {
  const titleLower = (title || '').toLowerCase();
  const isBytes = titleLower.includes('byte') || titleLower.includes('memory');

  // Handle bytes specially
  if (isBytes) {
    const absValue = Math.abs(value);
    if (absValue >= 1e12) return `${(value / 1e12).toFixed(1)} TB`;
    if (absValue >= 1e9) return `${(value / 1e9).toFixed(1)} GB`;
    if (absValue >= 1e6) return `${(value / 1e6).toFixed(1)} MB`;
    if (absValue >= 1e3) return `${(value / 1e3).toFixed(1)} KB`;
    return `${value.toFixed(0)} B`;
  }

  // Handle other large/small numbers
  const absValue = Math.abs(value);
  if (absValue >= 1e12) return `${(value / 1e12).toFixed(1)}T`;
  if (absValue >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
  if (absValue >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (absValue >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  if (absValue >= 1) return value.toFixed(2);
  if (absValue >= 0.01) return value.toFixed(3);
  if (absValue >= 0.0001) return value.toFixed(4);
  if (absValue === 0) return '0';
  return value.toExponential(1);
}

/**
 * Convert snake_case or kebab-case to Title Case
 * Example: "train_loss" -> "Train Loss"
 *
 * @param {string} str - String to convert
 * @returns {string} Title cased string
 */
export function toTitleCase(str) {
  if (!str) return '';
  return str
    .split(/[_-]/)
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}
