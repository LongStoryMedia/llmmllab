

/**
 * Text block that a user contributed to the thread.
 */
export interface UserMessageInputText {
  /**
   * Plain-text content supplied by the user.
   */
  text: string;
  /**
   * Type discriminator that is always `input_text`.
   */
  type: 'input_text';
}