

/**
 * A text output from the model.

 */
export interface EvalItemContentOutputText {
  /**
   * The text output from the model.

   */
  text: string;
  /**
   * The type of the output text. Always `output_text`.

   */
  type: 'output_text';
}