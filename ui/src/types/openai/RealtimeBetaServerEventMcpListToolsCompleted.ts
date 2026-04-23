

/**
 * Returned when listing MCP tools has completed for an item.
 */
export interface RealtimeBetaServerEventMCPListToolsCompleted {
  /**
   * The unique ID of the server event.
   */
  event_id: string;
  /**
   * The ID of the MCP list tools item.
   */
  item_id: string;
  /**
   * The event type, must be `mcp_list_tools.completed`.
   */
  type: string;
}