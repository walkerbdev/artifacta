/**
 * Debug logger that captures console logs and saves to file
 * Enabled via VITE_DEBUG_LOGS=true environment variable
 */

/**
 * DebugLogger class that captures console logs and saves them to disk.
 */
class DebugLogger {
  /**
   * Creates a new DebugLogger instance.
   */
  constructor() {
    this.logs = [];
    this.enabled = import.meta.env.VITE_DEBUG_LOGS === 'true';
    this.maxLogs = 10000; // Prevent memory overflow
    this.startTime = Date.now();

    if (this.enabled) {
      this.interceptConsole();
      this.startAutoWrite();
      console.log('[DebugLogger] Enabled - logs will be auto-saved to logs/ directory every 3s');
    }
  }

  /**
   * Intercepts console methods to capture logs.
   */
  interceptConsole() {
    const self = this;
    const methods = ['log', 'warn', 'error', 'info', 'debug'];

    methods.forEach(method => {
      const original = console[method];
      /**
       * Wrapper function for console methods.
       * @param {...unknown} args - Console method arguments
       */
      console[method] = function(...args) {
        // Filter out React Flow and React DevTools warnings
        const message = args[0];
        const shouldFilter = typeof message === 'string' && (
          message.includes('React Flow') ||
          message.includes('React DevTools')
        );

        if (!shouldFilter) {
          // Call original
          original.apply(console, args);

          // Capture to buffer
          const timestamp = Date.now() - self.startTime;
          const logEntry = {
            timestamp,
            level: method,
            message: args.map(arg => {
              if (typeof arg === 'object') {
                try {
                  return JSON.stringify(arg, null, 2);
                } catch {
                  return String(arg);
                }
              }
              return String(arg);
            }).join(' ')
          };

          self.logs.push(logEntry);

          // Trim if too many logs
          if (self.logs.length > self.maxLogs) {
            self.logs.shift();
          }
        }
      };
    });
  }

  /**
   * Starts automatic writing of logs to disk every 3 seconds.
   */
  startAutoWrite() {
    // Write every 3 seconds
    /**
     * Interval callback to write logs to disk.
     */
    setInterval(() => {
      this.writeToDisk();
    }, 3000);

    // Also write before page unload
    window.addEventListener('beforeunload', () => {
      this.writeToDisk();
    });
  }

  /**
   * Writes captured logs to disk via API endpoint.
   */
  writeToDisk() {
    if (this.logs.length === 0) return;

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `debug-${timestamp}.txt`;

    const content = this.logs.map(entry => {
      const time = (entry.timestamp / 1000).toFixed(3);
      return `[${time}s] [${entry.level.toUpperCase()}] ${entry.message}`;
    }).join('\n');

    // Write to logs directory via backend
    const apiUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

    fetch(`${apiUrl}/api/debug-logs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename, content })
    })
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      console.log(`[DebugLogger] Wrote ${this.logs.length} logs to ${data.path}`);
    })
    .catch(err => {
      console.error('[DebugLogger] Failed to write logs:', err);
    });
  }

  /**
   * Downloads logs as a text file to the browser.
   */
  downloadLogs() {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `artifacta-logs-${timestamp}.txt`;

    const content = this.logs.map(entry => {
      const time = (entry.timestamp / 1000).toFixed(3);
      return `[${time}s] [${entry.level.toUpperCase()}] ${entry.message}`;
    }).join('\n');

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    console.log(`[DebugLogger] Downloaded ${this.logs.length} logs to ${filename}`);
  }

  /**
   * Static method to download logs programmatically from global instance.
   */
  static download() {
    if (window.__debugLogger) {
      window.__debugLogger.downloadLogs();
    } else {
      console.warn('[DebugLogger] Not enabled. Start UI with --debug-logs flag');
    }
  }
}

// Initialize and expose globally
if (import.meta.env.VITE_DEBUG_LOGS === 'true') {
  window.__debugLogger = new DebugLogger();
  /**
   * Global function to download debug logs
   * @returns {void}
   */
  window.downloadLogs = () => DebugLogger.download();
}
