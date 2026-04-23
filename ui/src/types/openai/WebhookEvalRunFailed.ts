

/**
 * Event data payload.

 */
export interface Data {
  /**
   * The unique ID of the eval run.

   */
  id: string;
}

/**
 * Sent when an eval run has failed.

 */
export interface WebhookEvalRunFailed {
  /**
   * The Unix timestamp (in seconds) of when the eval run failed.

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
   * The type of the event. Always `eval.run.failed`.

   */
  type: 'eval.run.failed';
}