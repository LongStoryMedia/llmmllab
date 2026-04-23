

/**
 * Returned when listing MCP tools is in progress for an item.
 */
export interface RealtimeServerEventMCPListToolsInProgress {
  /**
   * The unique ID of the server event.
   */
  event_id: string;
  /**
   * The ID of the MCP list tools item.
   */
  item_id: string;
  /**
   * The event type, must be `mcp_list_tools.in_progress`.
   */
  type: string;
}