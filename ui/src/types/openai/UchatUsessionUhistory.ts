

/**
 * History retention preferences returned for the session.
 */
export interface ChatSessionHistory {
  /**
   * Indicates if chat history is persisted for the session.
   */
  enabled: boolean;
  recent_threads: number | unknown;
}