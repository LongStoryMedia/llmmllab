

/**
 * Confirmation payload returned after deleting a role.
 */
export interface RoleDeletedResource {
  /**
   * Whether the role was deleted.
   */
  deleted: boolean;
  /**
   * Identifier of the deleted role.
   */
  id: string;
  /**
   * Always `role.deleted`.
   */
  object: 'role.deleted';
}