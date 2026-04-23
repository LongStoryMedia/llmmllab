

/**
 * Fine-tuning job event object
 */
export interface FineTuningJobEvent {
  /**
   * The Unix timestamp (in seconds) for when the fine-tuning job was created.
   */
  created_at: number;
  /**
   * The data associated with the event.
   */
  data?: Record<string, unknown>;
  /**
   * The object identifier.
   */
  id: string;
  /**
   * The log level of the event.
   */
  level: 'info' | 'warn' | 'error';
  /**
   * The message of the event.
   */
  message: string;
  /**
   * The object type, which is always "fine_tuning.job.event".
   */
  object: 'fine_tuning.job.event';
  /**
   * The type of event.
   */
  type?: 'message' | 'metrics';
}