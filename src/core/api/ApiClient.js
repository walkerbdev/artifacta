/**
 * Centralized API Client
 *
 * Eliminates duplicate fetch/try-catch patterns across the codebase.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

/**
 * Centralized API client for backend communication.
 */
class ApiClient {
  /**
   * Creates a new ApiClient instance.
   * @param {string} baseUrl - Base URL for API requests
   */
  constructor(baseUrl = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  /**
   * Generic request method with centralized error handling.
   * @param {string} endpoint - API endpoint path
   * @param {object} options - Fetch options
   * @returns {Promise<object>} Response object
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
   * GET request returning JSON.
   * @param {string} endpoint - API endpoint path
   * @returns {Promise<object>} Parsed JSON response
   */
  async get(endpoint) {
    const response = await this.request(endpoint);
    return response.json();
  }

  /**
   * GET request returning raw Response (for CSV, images, etc.).
   * @param {string} endpoint - API endpoint path
   * @returns {Promise<object>} Raw Response object
   */
  async getRaw(endpoint) {
    return this.request(endpoint);
  }

  // ========== Endpoints Actually Used ==========

  /**
   * Get all runs with filters.
   * @param {object} options - Query options
   * @param {number} options.limit - Maximum number of runs to return
   * @param {boolean} options.includeTags - Include tags in response
   * @param {boolean} options.includeParams - Include parameters in response
   * @param {boolean} options.includeMetadata - Include metadata in response
   * @returns {Promise<Array<object>>} Array of run objects
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
   * Get single run with full details.
   * @param {string} runId - Run ID
   * @returns {Promise<object>} Run object with full details
   */
  async getRun(runId) {
    return this.get(`/api/runs/${runId}`);
  }

  /**
   * Get metrics for a run.
   * @param {string} runId - Run ID
   * @returns {Promise<object>} Metrics data
   */
  async getRunMetrics(runId) {
    return this.get(`/api/runs/${runId}/metrics`);
  }

  /**
   * Get artifacts for a run.
   * @param {string} runId - Run ID
   * @returns {Promise<Array<object>>} Array of artifact objects
   */
  async getArtifacts(runId) {
    return this.get(`/api/artifacts/${runId}`);
  }

  /**
   * Get artifact links (with input/output role) for a run.
   * @param {string} runId - Run ID
   * @returns {Promise<Array<object>>} Array of artifact link objects
   */
  async getArtifactLinks(runId) {
    return this.get(`/api/runs/${runId}/artifact-links`);
  }

  /**
   * Get artifact preview (raw response for CSV/JSON/image handling).
   * @param {string} artifactId - Artifact ID
   * @param {number} offset - Offset for pagination
   * @param {number} limit - Maximum number of items
   * @returns {Promise<object>} Raw response for artifact preview
   */
  async getArtifactPreview(artifactId, offset = 0, limit = 100) {
    const params = new URLSearchParams({
      offset: offset.toString(),
      limit: limit.toString()
    });
    return this.getRaw(`/api/artifact/${artifactId}/preview?${params}`);
  }

  /**
   * Get artifact download URL.
   * @param {string} artifactId - Artifact ID
   * @returns {string} Download URL for artifact
   */
  getArtifactDownloadUrl(artifactId) {
    return `${this.baseUrl}/api/artifact/${artifactId}/download`;
  }

  // ========== Project Endpoints ==========

  /**
   * Get all projects (explicit + implicit from runs).
   * @returns {Promise<Array<object>>} Array of project objects
   */
  async getProjects() {
    return this.get('/api/projects');
  }

  /**
   * Create a new project.
   * @param {string} projectId - Project ID
   * @returns {Promise<object>} Created project object
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
   * Get all notes for a project.
   * @param {string} projectId - Project ID
   * @returns {Promise<Array<object>>} Array of note objects
   */
  async getProjectNotes(projectId) {
    return this.get(`/api/projects/${projectId}/notes`);
  }

  /**
   * Get a specific note.
   * @param {string} projectId - Project ID
   * @param {string} noteId - Note ID
   * @returns {Promise<object>} Note object
   */
  async getProjectNote(projectId, noteId) {
    return this.get(`/api/projects/${projectId}/notes/${noteId}`);
  }

  /**
   * Create a new note.
   * @param {string} projectId - Project ID
   * @param {object} data - Note data
   * @returns {Promise<object>} Created note object
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
   * Update an existing note.
   * @param {string} projectId - Project ID
   * @param {string} noteId - Note ID
   * @param {object} data - Updated note data
   * @returns {Promise<object>} Updated note object
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
   * Delete a note.
   * @param {string} projectId - Project ID
   * @param {string} noteId - Note ID
   * @returns {Promise<void>}
   */
  async deleteProjectNote(projectId, noteId) {
    await this.request(`/api/projects/${projectId}/notes/${noteId}`, {
      method: 'DELETE'
    });
  }

  /**
   * Get attachments for a note.
   * @param {string} noteId - Note ID
   * @returns {Promise<Array<object>>} Array of attachment objects
   */
  async getNoteAttachments(noteId) {
    return this.get(`/api/notes/${noteId}/attachments`);
  }

  /**
   * Upload attachment to a note.
   * @param {string} noteId - Note ID
   * @param {File} file - File to upload
   * @returns {Promise<object>} Uploaded attachment object
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
   * Delete an attachment.
   * @param {string} attachmentId - Attachment ID
   * @returns {Promise<void>}
   */
  async deleteAttachment(attachmentId) {
    await this.request(`/api/attachments/${attachmentId}`, {
      method: 'DELETE'
    });
  }

  /**
   * Get attachment download URL.
   * @param {string} attachmentId - Attachment ID
   * @returns {string} Download URL for attachment
   */
  getAttachmentDownloadUrl(attachmentId) {
    return `${this.baseUrl}/api/attachments/${attachmentId}/download`;
  }
}

// Export singleton instance
export const apiClient = new ApiClient();
