

/**
 * Emitted when function-call arguments are finalized.
 */
export interface ResponseFunctionCallArgumentsDoneEvent {
  /**
   * The function-call arguments.
   */
  arguments: string;
  /**
   * The ID of the item.
   */
  item_id: string;
  /**
   * The name of the function that was called.
   */
  name: string;
  /**
   * The index of the output item.
   */
  output_index: number;
  /**
   * The sequence number of this event.
   */
  sequence_number: number;
  type: 'response.function_call_arguments.done';
}