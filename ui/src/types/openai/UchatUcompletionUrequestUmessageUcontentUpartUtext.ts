

/**
 * Learn about [text inputs](https://platform.openai.com/docs/guides/text-generation).

 */
export interface ChatCompletionRequestMessageContentPartText {
  /**
   * The text content.
   */
  text: string;
  /**
   * The type of the content part.
   */
  type: 'text';
}