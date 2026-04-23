

/**
 * Emitted when an MCP  tool call has failed.

 */
export interface ResponseMCPCallFailedEvent {
  /**
   * The ID of the MCP tool call item that failed.
   */
  item_id: string;
  /**
   * The index of the output item that failed.
   */
  output_index: number;
  /**
   * The sequence number of this event.
   */
  sequence_number: number;
  /**
   * The type of the event. Always 'response.mcp_call.failed'.
   */
  type: 'response.mcp_call.failed';
}