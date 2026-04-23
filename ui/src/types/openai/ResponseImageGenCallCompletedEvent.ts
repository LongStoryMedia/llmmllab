

/**
 * Emitted when an image generation tool call has completed and the final image is available.

 */
export interface ResponseImageGenCallCompletedEvent {
  /**
   * The unique identifier of the image generation item being processed.
   */
  item_id: string;
  /**
   * The index of the output item in the response's output array.
   */
  output_index: number;
  /**
   * The sequence number of this event.
   */
  sequence_number: number;
  /**
   * The type of the event. Always 'response.image_generation_call.completed'.
   */
  type: 'response.image_generation_call.completed';
}