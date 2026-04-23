

/**
 * Confirmation payload returned after deleting a thread.
 */
export interface DeletedThreadResource {
  /**
   * Indicates that the thread has been deleted.
   */
  deleted: boolean;
  /**
   * Identifier of the deleted thread.
   */
  id: string;
  /**
   * Type discriminator that is always `chatkit.thread.deleted`.
   */
  object: 'chatkit.thread.deleted';
}