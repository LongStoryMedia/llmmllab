

/**
 * Active per-minute request limit for the session.
 */
export interface ChatSessionRateLimits {
  /**
   * Maximum allowed requests per one-minute window.
   */
  max_requests_per_1_minute: number;
}