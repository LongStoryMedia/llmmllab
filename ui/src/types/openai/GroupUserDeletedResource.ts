

/**
 * Confirmation payload returned after removing a user from a group.
 */
export interface GroupUserDeletedResource {
  /**
   * Whether the group membership was removed.
   */
  deleted: boolean;
  /**
   * Always `group.user.deleted`.
   */
  object: 'group.user.deleted';
}