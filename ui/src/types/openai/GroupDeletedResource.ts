

/**
 * Confirmation payload returned after deleting a group.
 */
export interface GroupDeletedResource {
  /**
   * Whether the group was deleted.
   */
  deleted: boolean;
  /**
   * Identifier of the deleted group.
   */
  id: string;
  /**
   * Always `group.deleted`.
   */
  object: 'group.deleted';
}