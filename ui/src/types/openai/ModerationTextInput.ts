

/**
 * An object describing text to classify.
 */
export interface ModerationTextInput {
  /**
   * A string of text to classify.
   */
  text: string;
  /**
   * Always `text`.
   */
  type: 'text';
}