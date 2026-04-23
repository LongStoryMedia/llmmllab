

/**
 * Emitted when a file search is currently searching.
 */
export interface ResponseFileSearchCallSearchingEvent {
  /**
   * The ID of the output item that the file search call is initiated.

   */
  item_id: string;
  /**
   * The index of the output item that the file search call is searching.

   */
  output_index: number;
  /**
   * The sequence number of this event.
   */
  sequence_number: number;
  /**
   * The type of the event. Always `response.file_search_call.searching`.

   */
  type: 'response.file_search_call.searching';
}