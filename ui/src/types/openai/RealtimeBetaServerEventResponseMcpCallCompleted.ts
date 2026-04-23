

/**
 * Returned when an MCP tool call has completed successfully.
 */
export interface RealtimeBetaServerEventResponseMCPCallCompleted {
  /**
   * The unique ID of the server event.
   */
  event_id: string;
  /**
   * The ID of the MCP tool call item.
   */
  item_id: string;
  /**
   * The index of the output item in the response.
   */
  output_index: number;
  /**
   * The event type, must be `response.mcp_call.completed`.
   */
  type: string;
}