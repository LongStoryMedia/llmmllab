

/**
 * The aggregated completions usage details of the specific time bucket.
 */
export interface UsageCompletionsResult {
  api_key_id?: string | unknown;
  batch?: boolean | unknown;
  /**
   * The aggregated number of audio input tokens used, including cached tokens.
   */
  input_audio_tokens?: number;
  /**
   * The aggregated number of text input tokens that has been cached from previous requests. For customers subscribe to scale tier, this includes scale tier tokens.
   */
  input_cached_tokens?: number;
  /**
   * The aggregated number of text input tokens used, including cached tokens. For customers subscribe to scale tier, this includes scale tier tokens.
   */
  input_tokens: number;
  model?: string | unknown;
  /**
   * The count of requests made to the model.
   */
  num_model_requests: number;
  object: 'organization.usage.completions.result';
  /**
   * The aggregated number of audio output tokens used.
   */
  output_audio_tokens?: number;
  /**
   * The aggregated number of text output tokens used. For customers subscribe to scale tier, this includes scale tier tokens.
   */
  output_tokens: number;
  project_id?: string | unknown;
  service_tier?: string | unknown;
  user_id?: string | unknown;
}