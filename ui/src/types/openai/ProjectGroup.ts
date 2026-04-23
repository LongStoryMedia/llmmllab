

/**
 * Details about a group's membership in a project.
 */
export interface ProjectGroup {
  /**
   * Unix timestamp (in seconds) when the group was granted project access.
   */
  created_at: number;
  /**
   * Identifier of the group that has access to the project.
   */
  group_id: string;
  /**
   * Display name of the group.
   */
  group_name: string;
  /**
   * Always `project.group`.
   */
  object: 'project.group';
  /**
   * Identifier of the project.
   */
  project_id: string;
}