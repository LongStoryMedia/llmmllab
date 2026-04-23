

/**
 * Emitted when a reasoning summary text is completed.
 */
export interface ResponseReasoningSummaryTextDoneEvent {
  /**
   * The ID of the item this summary text is associated with.

   */
  item_id: string;
  /**
   * The index of the output item this summary text is associated with.

   */
  output_index: number;
  /**
   * The sequence number of this event.

   */
  sequence_number: number;
  /**
   * The index of the summary part within the reasoning summary.

   */
  summary_index: number;
  /**
   * The full text of the completed reasoning summary.

   */
  text: string;
  /**
   * The type of the event. Always `response.reasoning_summary_text.done`.

   */
  type: 'response.reasoning_summary_text.done';
}