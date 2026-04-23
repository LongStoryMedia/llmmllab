

/**
 * Controls request rate limits for the session.
 */
export interface RateLimitsParam {
  /**
   * Maximum number of requests allowed per minute for the session. Defaults to 10.
   */
  max_requests_per_1_minute?: number;
}