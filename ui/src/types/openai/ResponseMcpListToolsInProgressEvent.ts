

/**
 * Emitted when the system is in the process of retrieving the list of available MCP tools.

 */
export interface ResponseMCPListToolsInProgressEvent {
  /**
   * The ID of the MCP tool call item that is being processed.
   */
  item_id: string;
  /**
   * The index of the output item that is being processed.
   */
  output_index: number;
  /**
   * The sequence number of this event.
   */
  sequence_number: number;
  /**
   * The type of the event. Always 'response.mcp_list_tools.in_progress'.
   */
  type: 'response.mcp_list_tools.in_progress';
}