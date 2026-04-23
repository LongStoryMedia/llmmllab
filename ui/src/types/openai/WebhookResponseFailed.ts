

/**
 * Event data payload.

 */
export interface Data {
  /**
   * The unique ID of the model response.

   */
  id: string;
}

/**
 * Sent when a background response has failed.

 */
export interface WebhookResponseFailed {
  /**
   * The Unix timestamp (in seconds) of when the model response failed.

   */
  created_at: number;
  /**
   * Event data payload.

   */
  data: Data;
  /**
   * The unique ID of the event.

   */
  id: string;
  /**
   * The object of the event. Always `event`.

   */
  object?: 'event';
  /**
   * The type of the event. Always `response.failed`.

   */
  type: 'response.failed';
}