

/**
 * Represents an individual project.
 */
export interface Project {
  archived_at?: number | unknown;
  /**
   * The Unix timestamp (in seconds) of when the project was created.
   */
  created_at: number;
  /**
   * The identifier, which can be referenced in API endpoints
   */
  id: string;
  /**
   * The name of the project. This appears in reporting.
   */
  name: string;
  /**
   * The object type, which is always `organization.project`
   */
  object: 'organization.project';
  /**
   * `active` or `archived`
   */
  status: 'active' | 'archived';
}