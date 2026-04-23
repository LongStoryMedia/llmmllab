

/**
 * Emitted when a web search call is completed.
 */
export interface ResponseWebSearchCallCompletedEvent {
  /**
   * Unique ID for the output item associated with the web search call.

   */
  item_id: string;
  /**
   * The index of the output item that the web search call is associated with.

   */
  output_index: number;
  /**
   * The sequence number of the web search call being processed.
   */
  sequence_number: number;
  /**
   * The type of the event. Always `response.web_search_call.completed`.

   */
  type: 'response.web_search_call.completed';
}