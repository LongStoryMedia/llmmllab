

/**
 * The content part that is done.
 */
export interface Part {
  /**
   * Base64-encoded audio data (if type is "audio").
   */
  audio?: string;
  /**
   * The text content (if type is "text").
   */
  text?: string;
  /**
   * The transcript of the audio (if type is "audio").
   */
  transcript?: string;
  /**
   * The content type ("text", "audio").
   */
  type?: 'text' | 'audio';
}

/**
 * Returned when a content part is done streaming in an assistant message item.
Also emitted when a Response is interrupted, incomplete, or cancelled.

 */
export interface RealtimeBetaServerEventResponseContentPartDone {
  /**
   * The index of the content part in the item's content array.
   */
  content_index: number;
  /**
   * The unique ID of the server event.
   */
  event_id: string;
  /**
   * The ID of the item.
   */
  item_id: string;
  /**
   * The index of the output item in the response.
   */
  output_index: number;
  /**
   * The content part that is done.
   */
  part: Part;
  /**
   * The ID of the response.
   */
  response_id: string;
  /**
   * The event type, must be `response.content_part.done`.
   */
  type: string;
}