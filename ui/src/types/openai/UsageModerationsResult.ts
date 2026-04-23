

/**
 * The aggregated moderations usage details of the specific time bucket.
 */
export interface UsageModerationsResult {
  api_key_id?: string | unknown;
  /**
   * The aggregated number of input tokens used.
   */
  input_tokens: number;
  model?: string | unknown;
  /**
   * The count of requests made to the model.
   */
  num_model_requests: number;
  object: 'organization.usage.moderations.result';
  project_id?: string | unknown;
  user_id?: string | unknown;
}