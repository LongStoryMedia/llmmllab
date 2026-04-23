

/**
 * Emitted at the beginning of a Response to indicate the updated rate limits. 
When a Response is created some tokens will be "reserved" for the output 
tokens, the rate limits shown here reflect that reservation, which is then 
adjusted accordingly once the Response is completed.

 */
export interface RealtimeServerEventRateLimitsUpdated {
  /**
   * The unique ID of the server event.
   */
  event_id: string;
  /**
   * List of rate limit information.
   */
  rate_limits: (RateLimitsItem)[];
  /**
   * The event type, must be `rate_limits.updated`.
   */
  type: string;
}