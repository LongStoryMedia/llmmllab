

/**
 * Event data payload.

 */
export interface Data {
  /**
   * The unique ID of the fine-tuning job.

   */
  id: string;
}

/**
 * Sent when a fine-tuning job has succeeded.

 */
export interface WebhookFineTuningJobSucceeded {
  /**
   * The Unix timestamp (in seconds) of when the fine-tuning job succeeded.

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
   * The type of the event. Always `fine_tuning.job.succeeded`.

   */
  type: 'fine_tuning.job.succeeded';
}