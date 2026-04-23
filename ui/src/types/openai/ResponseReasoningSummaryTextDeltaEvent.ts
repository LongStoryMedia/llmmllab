

/**
 * Emitted when a delta is added to a reasoning summary text.
 */
export interface ResponseReasoningSummaryTextDeltaEvent {
  /**
   * The text delta that was added to the summary.

   */
  delta: string;
  /**
   * The ID of the item this summary text delta is associated with.

   */
  item_id: string;
  /**
   * The index of the output item this summary text delta is associated with.

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
   * The type of the event. Always `response.reasoning_summary_text.delta`.

   */
  type: 'response.reasoning_summary_text.delta';
}