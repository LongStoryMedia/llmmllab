

/**
 * The text content that is part of a message.
 */
export interface MessageRequestContentTextObject {
  /**
   * Text content to be sent to the model
   */
  text: string;
  /**
   * Always `text`.
   */
  type: 'text';
}