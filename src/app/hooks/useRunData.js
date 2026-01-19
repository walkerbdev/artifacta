import { useState, useEffect, useCallback } from 'react';
import { apiClient } from '@/core/api/ApiClient';

/**
 * Unified hook for fetching run data from REST API
 * Polls continuously to pick up new runs from database
 */
export const useRunData = () => {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [runCount, setRunCount] = useState(0);

  // Fetch all runs from REST API
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

  // Fetch metrics for a specific run (raw data only, UI does formatting)
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
