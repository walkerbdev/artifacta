/**
 * Filter console warnings from third-party libraries
 */
export function initConsoleFilter() {
  const originalWarn = console.warn;

  /**
   * Wrapper function to filter console warnings from React Flow and React DevTools.
   * @param {...unknown} args - Arguments passed to console.warn
   */
  console.warn = function(...args) {
    const message = args[0];
    const shouldFilter = typeof message === 'string' && (
      message.includes('React Flow') ||
      message.includes('React DevTools')
    );

    if (!shouldFilter) {
      originalWarn.apply(console, args);
    }
  };
}
