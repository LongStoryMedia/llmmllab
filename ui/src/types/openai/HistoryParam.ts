

/**
 * Controls how much historical context is retained for the session.
 */
export interface HistoryParam {
  /**
   * Enables chat users to access previous ChatKit threads. Defaults to true.
   */
  enabled?: boolean;
  /**
   * Number of recent ChatKit threads users have access to. Defaults to unlimited when unset.
   */
  recent_threads?: number;
}