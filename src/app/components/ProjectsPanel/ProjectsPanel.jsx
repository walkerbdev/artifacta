import React, { useState, useEffect } from 'react';
import { HiPlus, HiDocumentText } from 'react-icons/hi';
import { apiClient } from '@/core/api/ApiClient';
import { SIDEBAR_STYLES } from '@/app/styles/sidebarConstants';

export const ProjectsPanel = ({ runs, onNoteSelect, onNewNote, onTabChange }) => {
  const [projects, setProjects] = useState([]);
  const [selectedProject, setSelectedProject] = useState(null);
  const [notes, setNotes] = useState([]);
  const [notesVersion, setNotesVersion] = useState(0);

  // Fetch all projects (explicit + implicit from runs)
  useEffect(() => {
    const loadProjects = async () => {
      try {
        const data = await apiClient.getProjects();
        const allProjects = data.projects || [];

        // Create project map with runs
        const projectMap = {};

        // Add all explicit/implicit projects
        allProjects.forEach(proj => {
          projectMap[proj.project_id] = {
            id: proj.project_id,
            runs: [],
            is_implicit: proj.is_implicit
          };
        });

        // Add runs to projects
        if (runs && runs.length > 0) {
          runs.forEach(run => {
            const projectId = run.project || 'default';
            if (!projectMap[projectId]) {
              projectMap[projectId] = {
                id: projectId,
                runs: [],
                is_implicit: true
              };
            }
            projectMap[projectId].runs.push(run);
          });
        }

        const projectList = Object.values(projectMap);
        setProjects(projectList);

        // Auto-select first project only if no project is selected and we have projects
        if (projectList.length > 0 && selectedProject === null) {
          setSelectedProject(projectList[0].id);
        }
      } catch (error) {
        console.error('Failed to load projects:', error);
        setProjects([]);
      }
    };

    loadProjects();
  }, [runs, notesVersion, selectedProject]);

  // Load notes when project selected or when notesVersion changes
  useEffect(() => {
    if (selectedProject) {
      loadNotes(selectedProject);
    }
  }, [selectedProject, notesVersion]);

  // Listen for custom event to refresh notes
  useEffect(() => {
    const handleRefresh = () => {
      setNotesVersion(prev => prev + 1);
    };

    window.addEventListener('refreshProjectNotes', handleRefresh);
    return () => window.removeEventListener('refreshProjectNotes', handleRefresh);
  }, []);

  const loadNotes = async (projectId) => {
    try {
      const data = await apiClient.getProjectNotes(projectId);
      setNotes(data.notes || []);
    } catch (error) {
      console.error('Failed to load notes:', error);
      setNotes([]);
    }
  };

  const handleProjectClick = (projectId) => {
    if (selectedProject === projectId) {
      setSelectedProject(null);
      setNotes([]);
    } else {
      setSelectedProject(projectId);
    }
  };

  const handleNoteClick = (note) => {
    // Switch to Notes tab and load the note
    onTabChange('notes');
    onNoteSelect(selectedProject, note);
  };

  const handleCreateNote = () => {
    // Switch to Notes tab and create new note
    onTabChange('notes');
    onNewNote(selectedProject);
  };

  const handleCreateProject = async () => {
    const projectName = window.prompt('Enter project name:');
    if (!projectName) return;

    try {
      await apiClient.createProject(projectName);
      // Refresh projects list
      setNotesVersion(prev => prev + 1);
    } catch (error) {
      window.alert('Failed to create project: ' + error.message);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
      {projects.map(project => {
        const isSelected = selectedProject === project.id;
        const projectNotes = isSelected ? notes : [];

        return (
          <div key={project.id} style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
            {/* Project Header */}
            <div
              onClick={() => handleProjectClick(project.id)}
              style={{
                padding: '6px 8px',
                fontSize: '0.75rem',
                color: 'black',
                cursor: 'pointer',
                userSelect: 'none',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                background: isSelected ? 'white' : 'transparent',
                borderRadius: '4px',
                transition: 'background 0.2s'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'white';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = isSelected ? 'white' : 'transparent';
              }}
            >
              <span style={{ fontSize: '10px', color: 'black' }}>
                {isSelected ? '▼' : '▶'}
              </span>
              <span style={{
                fontFamily: SIDEBAR_STYLES.fontFamily,
                fontSize: SIDEBAR_STYLES.fontSize.header,
                fontWeight: SIDEBAR_STYLES.fontWeight.header
              }}>
                {project.id}
              </span>
            </div>

            {/* Notes list */}
            {isSelected && (
              <div style={{ paddingLeft: '16px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                {/* Notes */}
                {projectNotes.map(note => (
                  <div
                    key={note.id}
                    onClick={() => handleNoteClick(note)}
                    style={{
                      padding: '8px',
                      fontSize: SIDEBAR_STYLES.fontSize.item,
                      borderRadius: '4px',
                      background: 'white',
                      cursor: 'pointer',
                      transition: 'all 0.2s',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = '#f9fafb';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = 'white';
                    }}
                  >
                    <HiDocumentText size={14} color="#6b7280" />
                    <span style={{
                      flex: 1,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                      fontFamily: SIDEBAR_STYLES.fontFamily,
                      fontWeight: SIDEBAR_STYLES.fontWeight.item
                    }}>
                      {note.title}
                    </span>
                  </div>
                ))}

                {/* New Note button */}
                <button
                  onClick={handleCreateNote}
                  style={{
                    padding: '6px 12px',
                    fontSize: SIDEBAR_STYLES.fontSize.button,
                    background: '#f3f4f6',
                    color: '#374151',
                    border: '1px solid #e5e7eb',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px',
                    transition: 'all 0.2s',
                    fontWeight: 500,
                    marginRight: '8px'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = '#e5e7eb';
                    e.currentTarget.style.borderColor = '#d1d5db';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = '#f3f4f6';
                    e.currentTarget.style.borderColor = '#e5e7eb';
                  }}
                >
                  <HiPlus size={12} /> New Notebook
                </button>
              </div>
            )}
          </div>
        );
      })}

      {/* New Project button - at the end */}
      <button
        onClick={handleCreateProject}
        style={{
          padding: '6px 12px',
          fontSize: SIDEBAR_STYLES.fontSize.button,
          background: '#f3f4f6',
          color: '#374151',
          border: '1px solid #e5e7eb',
          borderRadius: '4px',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
          transition: 'all 0.2s',
          fontWeight: 500,
          marginLeft: '8px',
          marginRight: '8px',
          marginTop: '4px',
          marginBottom: '8px'
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.background = '#e5e7eb';
          e.currentTarget.style.borderColor = '#d1d5db';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.background = '#f3f4f6';
          e.currentTarget.style.borderColor = '#e5e7eb';
        }}
      >
        <HiPlus size={12} /> New Project
      </button>
    </div>
  );
};
