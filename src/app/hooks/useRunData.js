import { useState, useEffect, useCallback } from 'react';
import { apiClient } from '@/core/api/ApiClient';

/**
 * Custom hook for fetching and polling experiment run data from REST API
 *
 * Central data-fetching hook used throughout the application. Loads all runs from
 * the backend and continuously polls for new runs (e.g., from ongoing experiments).
 *
 * Features:
 * - Initial fetch on mount with loading state
 * - Continuous polling every 2 seconds for new runs
 * - Smart re-render prevention (only updates when run list changes)
 * - Stable reference if runs unchanged (prevents downstream re-renders)
 * - Error handling with error state
 * - Run count tracking
 * - Individual run metrics fetching
 *
 * Why polling:
 * - Detects newly completed experiments without page refresh
 * - Simple alternative to WebSocket/SSE for small-scale deployments
 * - 2-second interval balances freshness vs server load
 *
 * Re-render optimization:
 * - Compares run_id lists, not full object deep equality
 * - Returns previous reference if runs unchanged
 * - Critical for components with `runs` in dependencies (LineageTab, etc.)
 *
 * @returns {object} Run data and utilities:
 *   - runs: Array<object> - All experiment runs
 *   - loading: boolean - Initial loading state
 *   - error: string|null - Error message if fetch failed
 *   - runCount: number - Total number of runs
 *   - fetchRuns: function - Manually trigger fetch (isInitialFetch: boolean)
 *   - fetchRunMetrics: function - Fetch metrics for specific run (runId: string)
 *
 * @example
 * const { runs, loading, error, fetchRuns } = useRunData();
 *
 * if (loading) return <div>Loading...</div>;
 * if (error) return <div>Error: {error}</div>;
 *
 * return <RunList runs={runs} onRefresh={() => fetchRuns(true)} />;
 */
export const useRunData = () => {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [runCount, setRunCount] = useState(0);

  /**
   * Fetch all runs from REST API.
   * @param {boolean} isInitialFetch - Whether this is the initial fetch
   * @returns {Promise<void>}
   */
  const fetchRuns = useCallback(async (isInitialFetch = false) => {
    try {
      // Only show loading on initial fetch, not on polls
      if (isInitialFetch) {
        setLoading(true);
      }
      setError(null);
      const allRuns = await apiClient.getRuns();

      // Only update if data actually changed (prevent unnecessary re-renders)
      setRuns(prevRuns => {
        // Fast length check first
        if (prevRuns.length !== allRuns.length) {
          setRunCount(allRuns.length);
          return allRuns;
        }

        // Compare run_ids only (stable check)
        // Don't trigger re-render for stream/metric updates that happen during training
        const prevIds = prevRuns.map(r => r.run_id).join(',');
        const newIds = allRuns.map(r => r.run_id).join(',');

        if (prevIds !== newIds) {
          setRunCount(allRuns.length);
          return allRuns;
        }

        // Same runs exist - keep previous reference to prevent downstream re-renders
        // This is critical for components like LineageTab that use allRuns in dependencies
        return prevRuns;
      });
    } catch (err) {
      console.error('Failed to fetch runs:', err);
      setError(err.toString());
    } finally {
      if (isInitialFetch) {
        setLoading(false);
      }
    }
  }, []);

  /**
   * Fetch metrics for a specific run (raw data only, UI does formatting).
   * @param {string} runId - Run ID to fetch metrics for
   * @returns {Promise<object>} Metrics data for the run
   */
  const fetchRunMetrics = useCallback(async (runId) => {
    try {
      return await apiClient.getRunMetrics(runId);
    } catch (err) {
      console.error(`Failed to fetch metrics for run ${runId}:`, err);
      throw err;
    }
  }, []);

  // Initial fetch on mount
  useEffect(() => {
    fetchRuns(true);
  }, [fetchRuns]);

  // Poll to pick up newly completed runs from database
  // Runs are immutable, but we need to detect when NEW runs are written
  useEffect(() => {
    // Poll every 2 seconds to pick up new runs
    const pollInterval = setInterval(() => {
      fetchRuns(false);
    }, 2000);

    return () => clearInterval(pollInterval);
  }, [fetchRuns]);

  return {
    runs,
    loading,
    error,
    runCount,
    fetchRuns,
    fetchRunMetrics,
  };
};
