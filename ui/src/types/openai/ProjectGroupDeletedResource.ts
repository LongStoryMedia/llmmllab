

/**
 * Confirmation payload returned after removing a group from a project.
 */
export interface ProjectGroupDeletedResource {
  /**
   * Whether the group membership in the project was removed.
   */
  deleted: boolean;
  /**
   * Always `project.group.deleted`.
   */
  object: 'project.group.deleted';
}