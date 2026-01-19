import React, { useState, useMemo } from 'react';
import { Toaster } from 'react-hot-toast';
import 'katex/dist/katex.min.css';
import './Workspace.scss';
import { RunSelector } from '@/app/components';
import { useRunData } from '@/app/hooks';
import { Sidebar, TabbedInterface } from '@/app/components/layout';
import { CollapsibleSection } from './CollapsibleSection';
import { discoverPlots } from '@/app/utils/plotDiscovery';
import { PlotSection } from './PlotSection';
import { LineageTab } from '@/app/components/LineageTab/LineageTab';
import { ArtifactTab } from '@/app/components/ArtifactTab/ArtifactTab';
import { RunTree } from '@/app/components/RunTree';
import { SweepsTab } from '@/app/components/SweepsTab';
import { ChatTab } from '@/app/components/ChatTab/ChatTab';
import { ProjectNotesTab } from '@/app/components/ProjectNotesTab/ProjectNotesTab';
import { ProjectsPanel } from '@/app/components/ProjectsPanel/ProjectsPanel';
import { ArtifactsPanel } from '@/app/components/ArtifactsPanel';

const Workspace = () => {
  // Sidebar state
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);

  // Tab state - which tabs are visible and which is active
  const [visibleTabs, setVisibleTabs] = useState(['notes', 'visualizations', 'tables', 'sweeps', 'lineage', 'data', 'chat']);
  const [activeTab, setActiveTab] = useState('notes');

  // Multi-run comparison state
  const [selectedRunIds, setSelectedRunIds] = useState([]);

  // Selected artifact for Artifact tab (set from Lineage tab or ArtifactsPanel)
  const [selectedArtifact, setSelectedArtifact] = useState(null);

  // Project notes state - for sidebar to Notes tab communication
  const [selectedNoteData, setSelectedNoteData] = useState(null);

  // Fetch run history from database
  const { runs, loading: runsLoading } = useRunData();

  return (
    <>
    <Toaster />
    <div className="viz-workspace-container">
      {/* Collapsible Sidebar with Controls */}
      <Sidebar isCollapsed={isSidebarCollapsed} onToggle={() => setIsSidebarCollapsed(!isSidebarCollapsed)}>
      {/* Projects */}
      <CollapsibleSection
        title="Projects"
        defaultCollapsed={true}
      >
        <ProjectsPanel
          runs={runs}
          onNoteSelect={(projectId, note) => {
            setSelectedNoteData({ projectId, noteId: note.id, isCreating: false });
          }}
          onNewNote={(projectId) => {
            setSelectedNoteData({ projectId, noteId: null, isCreating: true });
          }}
          onTabChange={setActiveTab}
        />
      </CollapsibleSection>

      {/* Runs - Git-tree browser grouped by hash.code */}
      <CollapsibleSection
        title="Runs"
        defaultCollapsed={true}
      >
        {runs && runs.length > 0 && (
          <RunTree
            runs={runs}
            selectedRunIds={selectedRunIds}
            onRunSelectionChange={setSelectedRunIds}
          />
        )}
      </CollapsibleSection>

      {/* Files - File browser for selected runs */}
      <CollapsibleSection
        title="Files"
        defaultCollapsed={true}
      >
        <ArtifactsPanel
          selectedRunIds={selectedRunIds}
          onFileSelect={(fileData) => {
            setSelectedArtifact(fileData);
            setActiveTab('data');
          }}
        />
      </CollapsibleSection>
        </Sidebar>

      {/* Main Content Area */}
      <div className="viz-workspace-main">
        {/* Tabbed Interface for main panels */}
        <TabbedInterface
          tabs={[
            {
              id: 'notes',
              label: 'Notebooks',
              content: (
                <ProjectNotesTab
                  projectId={runs?.[0]?.project || 'default'}
                  availableRuns={runs || []}
                  selectedRunIds={selectedRunIds}
                  externalNoteId={selectedNoteData?.noteId}
                  externalProjectId={selectedNoteData?.projectId}
                  isCreatingNew={selectedNoteData?.isCreating}
                />
              )
            },
            {
              id: 'visualizations',
              label: 'Plots',
              content: (
                <>
                  {/* Filter Controls */}

                  <div className="viz-viz-grid">
                  {/* Dynamically render visualizations based on available data */}
                  {useMemo(() => {
                    if (!runs || runs.length === 0 || selectedRunIds.length === 0) {
                      return <div className="viz-viz-row"></div>;
                    }

                    // Get selected runs
                    const selectedRuns = runs.filter(r => selectedRunIds.includes(r.run_id));

                    // Merge structured_data from all selected runs
                    // Strategy: Group 'series' and 'curve' primitives by name AND section for multi-run overlay
                    //           Keep other primitives separate to avoid visual clutter
                    const mergedStructuredData = {};
                    selectedRuns.forEach(run => {
                      if (run.structured_data) {
                        Object.entries(run.structured_data).forEach(([name, entries]) => {
                          const latestEntry = entries[entries.length - 1];
                          const primitiveType = latestEntry.primitive_type;
                          const section = latestEntry.section || 'General';

                          // Primitives that should be grouped across runs (overlay on same plot)
                          const shouldGroup = primitiveType === 'series' || primitiveType === 'curve';

                          // Determine key: group by name+section for series/curve, prefix for others
                          let key;
                          if (shouldGroup) {
                            // Group by BOTH name and section to avoid mixing primitives from different sections
                            key = `${section}::${name}`;
                          } else if (selectedRuns.length > 1) {
                            key = `${run.name || run.run_id.substring(0, 8)}: ${name}`;
                          } else {
                            key = name;
                          }

                          // For grouped primitives, accumulate data from all runs
                          if (shouldGroup) {
                            if (!mergedStructuredData[key]) {
                              mergedStructuredData[key] = [];
                            }
                            // Add run metadata to each entry for later identification
                            entries.forEach(entry => {
                              mergedStructuredData[key].push({
                                ...entry,
                                _runName: run.name || run.run_id.substring(0, 8),
                                _runId: run.run_id
                              });
                            });
                          } else {
                            // For non-grouped primitives, keep separate
                            mergedStructuredData[key] = entries;
                          }
                        });
                      }
                    });

                    if (Object.keys(mergedStructuredData).length === 0) {
                      return null;
                    }

                    // Discover plots from merged structured_data, grouped by section
                    const plotsBySection = discoverPlots(mergedStructuredData);

                    if (Object.keys(plotsBySection).length === 0) {
                      return <div className="viz-viz-row">No plots discovered</div>;
                    }

                    // Render each section with draggable plots
                    return (
                      <>
                        {Object.entries(plotsBySection).map(([sectionName, plots]) => (
                          <PlotSection
                            key={sectionName}
                            sectionName={sectionName}
                            plots={plots}
                          />
                        ))}
                      </>
                    );
                  }, [runs, selectedRunIds])}
                </div>
                </>
              )
            },
            {
              id: 'tables',
              label: 'Tables',
              content: (
                <RunSelector
                  selectedRunIds={selectedRunIds}
                  onRunSelectionChange={setSelectedRunIds}
                  runs={runs}
                  runsLoading={runsLoading}
                />
              )
            },
            {
              id: 'sweeps',
              label: 'Sweeps',
              content: (
                <SweepsTab
                  runs={runs}
                  selectedRunIds={selectedRunIds}
                />
              )
            },
            {
              id: 'lineage',
              label: 'Lineage',
              content: (
                <LineageTab
                  selectedRunIds={selectedRunIds}
                  allRuns={runs}
                  onDatasetSelect={(artifact) => {
                    setSelectedArtifact(artifact);
                    setActiveTab('data');
                  }}
                  onArtifactView={(artifact) => {
                    setSelectedArtifact(artifact);
                    setActiveTab('data');
                  }}
                />
              )
            },
            {
              id: 'data',
              label: 'Artifacts',
              content: (
                <ArtifactTab selectedArtifact={selectedArtifact} />
              )
            },
            {
              id: 'chat',
              label: 'Chat',
              content: (
                <ChatTab selectedRunIds={selectedRunIds} allRuns={runs} />
              )
            }
          ]}
          visibleTabs={visibleTabs}
          onTabVisibilityChange={(tabId, isVisible) => {
            setVisibleTabs(prev =>
              isVisible ? [...prev, tabId] : prev.filter(id => id !== tabId)
            );
          }}
          activeTab={activeTab}
          onTabChange={setActiveTab}
        />
      </div>
    </div>
  </>
  );
};

export default Workspace;
