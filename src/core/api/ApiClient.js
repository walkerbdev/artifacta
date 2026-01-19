/**
 * Centralized API Client
 *
 * Eliminates duplicate fetch/try-catch patterns across the codebase.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL;

class ApiClient {
  constructor(baseUrl = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  /**
   * Generic request method with centralized error handling
   */
  async request(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    const response = await fetch(url, options);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    return response;
  }

  /**
   * GET request returning JSON
   */
  async get(endpoint) {
    const response = await this.request(endpoint);
    return response.json();
  }

  /**
   * GET request returning raw Response (for CSV, images, etc.)
   */
  async getRaw(endpoint) {
    return this.request(endpoint);
  }

  // ========== Endpoints Actually Used ==========

  /**
   * Get all runs with filters
   */
  async getRuns(options = {}) {
    const {
      limit = 100,
      includeTags = true,
      includeParams = true,
      includeMetadata = true,
    } = options;

    const params = new URLSearchParams({ limit: limit.toString() });
    if (includeTags) params.append('include_tags', 'true');
    if (includeParams) params.append('include_params', 'true');
    if (includeMetadata) params.append('include_metadata', 'true');

    return this.get(`/api/runs?${params}`);
  }

  /**
   * Get single run with full details
   */
  async getRun(runId) {
    return this.get(`/api/runs/${runId}`);
  }

  /**
   * Get metrics for a run
   */
  async getRunMetrics(runId) {
    return this.get(`/api/runs/${runId}/metrics`);
  }

  /**
   * Get artifacts for a run
   */
  async getArtifacts(runId) {
    return this.get(`/api/artifacts/${runId}`);
  }

  /**
   * Get artifact links (with input/output role) for a run
   */
  async getArtifactLinks(runId) {
    return this.get(`/api/runs/${runId}/artifact-links`);
  }

  /**
   * Get artifact preview (raw response for CSV/JSON/image handling)
   */
  async getArtifactPreview(artifactId, offset = 0, limit = 100) {
    const params = new URLSearchParams({
      offset: offset.toString(),
      limit: limit.toString()
    });
    return this.getRaw(`/api/artifact/${artifactId}/preview?${params}`);
  }

  /**
   * Get artifact download URL
   */
  getArtifactDownloadUrl(artifactId) {
    return `${this.baseUrl}/api/artifact/${artifactId}/download`;
  }

  // ========== Project Endpoints ==========

  /**
   * Get all projects (explicit + implicit from runs)
   */
  async getProjects() {
    return this.get('/api/projects');
  }

  /**
   * Create a new project
   */
  async createProject(projectId) {
    const response = await this.request('/api/projects', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project_id: projectId })
    });
    return response.json();
  }

  // ========== Project Notes Endpoints ==========

  /**
   * Get all notes for a project
   */
  async getProjectNotes(projectId) {
    return this.get(`/api/projects/${projectId}/notes`);
  }

  /**
   * Get a specific note
   */
  async getProjectNote(projectId, noteId) {
    return this.get(`/api/projects/${projectId}/notes/${noteId}`);
  }

  /**
   * Create a new note
   */
  async createProjectNote(projectId, data) {
    const response = await this.request(`/api/projects/${projectId}/notes`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    return response.json();
  }

  /**
   * Update an existing note
   */
  async updateProjectNote(projectId, noteId, data) {
    const response = await this.request(`/api/projects/${projectId}/notes/${noteId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    return response.json();
  }

  /**
   * Delete a note
   */
  async deleteProjectNote(projectId, noteId) {
    await this.request(`/api/projects/${projectId}/notes/${noteId}`, {
      method: 'DELETE'
    });
  }

  /**
   * Get attachments for a note
   */
  async getNoteAttachments(noteId) {
    return this.get(`/api/notes/${noteId}/attachments`);
  }

  /**
   * Upload attachment to a note
   */
  async uploadAttachment(noteId, file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await this.request(`/api/notes/${noteId}/attachments`, {
      method: 'POST',
      body: formData
    });
    return response.json();
  }

  /**
   * Delete an attachment
   */
  async deleteAttachment(attachmentId) {
    await this.request(`/api/attachments/${attachmentId}`, {
      method: 'DELETE'
    });
  }

  /**
   * Get attachment download URL
   */
  getAttachmentDownloadUrl(attachmentId) {
    return `${this.baseUrl}/api/attachments/${attachmentId}/download`;
  }
}

// Export singleton instance
export const apiClient = new ApiClient();
