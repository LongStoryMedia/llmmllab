

/**
 * Confirmation payload returned after adding a user to a group.
 */
export interface GroupUserAssignment {
  /**
   * Identifier of the group the user was added to.
   */
  group_id: string;
  /**
   * Always `group.user`.
   */
  object: 'group.user';
  /**
   * Identifier of the user that was added.
   */
  user_id: string;
}