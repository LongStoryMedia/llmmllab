

/**
 * Returned when listing MCP tools has failed for an item.
 */
export interface RealtimeBetaServerEventMCPListToolsFailed {
  /**
   * The unique ID of the server event.
   */
  event_id: string;
  /**
   * The ID of the MCP list tools item.
   */
  item_id: string;
  /**
   * The event type, must be `mcp_list_tools.failed`.
   */
  type: string;
}