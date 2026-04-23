

/**
 * The request counts for different statuses within the batch.
 */
export interface BatchRequestCounts {
  /**
   * Number of requests that have been completed successfully.
   */
  completed: number;
  /**
   * Number of requests that have failed.
   */
  failed: number;
  /**
   * Total number of requests in the batch.
   */
  total: number;
}