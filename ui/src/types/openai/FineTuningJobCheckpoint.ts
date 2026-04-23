

/**
 * Metrics at the step number during the fine-tuning job.
 */
export interface Metrics {
  full_valid_loss?: number;
  full_valid_mean_token_accuracy?: number;
  step?: number;
  train_loss?: number;
  train_mean_token_accuracy?: number;
  valid_loss?: number;
  valid_mean_token_accuracy?: number;
}

/**
 * The `fine_tuning.job.checkpoint` object represents a model checkpoint for a fine-tuning job that is ready to use.

 */
export interface FineTuningJobCheckpoint {
  /**
   * The Unix timestamp (in seconds) for when the checkpoint was created.
   */
  created_at: number;
  /**
   * The name of the fine-tuned checkpoint model that is created.
   */
  fine_tuned_model_checkpoint: string;
  /**
   * The name of the fine-tuning job that this checkpoint was created from.
   */
  fine_tuning_job_id: string;
  /**
   * The checkpoint identifier, which can be referenced in the API endpoints.
   */
  id: string;
  /**
   * Metrics at the step number during the fine-tuning job.
   */
  metrics: Metrics;
  /**
   * The object type, which is always "fine_tuning.job.checkpoint".
   */
  object: 'fine_tuning.job.checkpoint';
  /**
   * The step number that the checkpoint was created at.
   */
  step_number: number;
}