

/**
 * A summary text from the model.
 */
export interface Summary {
  /**
   * A summary of the reasoning output from the model so far.
   */
  text: string;
  /**
   * The type of the object. Always `summary_text`.
   */
  type: 'summary_text';
}