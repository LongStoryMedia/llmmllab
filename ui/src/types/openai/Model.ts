

/**
 * Describes an OpenAI model offering that can be used with the API.
 */
export interface Model {
  /**
   * The Unix timestamp (in seconds) when the model was created.
   */
  created: number;
  /**
   * The model identifier, which can be referenced in the API endpoints.
   */
  id: string;
  /**
   * The object type, which is always "model".
   */
  object: 'model';
  /**
   * The organization that owns the model.
   */
  owned_by: string;
}