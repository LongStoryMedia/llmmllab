

/**
 * Indicates that a thread has been closed.
 */
export interface ClosedStatus {
  reason: string | unknown;
  /**
   * Status discriminator that is always `closed`.
   */
  type: 'closed';
}