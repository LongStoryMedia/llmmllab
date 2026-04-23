

/**
 * Emitted when refusal text is finalized.
 */
export interface ResponseRefusalDoneEvent {
  /**
   * The index of the content part that the refusal text is finalized.

   */
  content_index: number;
  /**
   * The ID of the output item that the refusal text is finalized.

   */
  item_id: string;
  /**
   * The index of the output item that the refusal text is finalized.

   */
  output_index: number;
  /**
   * The refusal text that is finalized.

   */
  refusal: string;
  /**
   * The sequence number of this event.

   */
  sequence_number: number;
  /**
   * The type of the event. Always `response.refusal.done`.

   */
  type: 'response.refusal.done';
}