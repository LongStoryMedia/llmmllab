

/**
 * Returned when MCP tool call arguments are updated during response generation.
 */
export interface RealtimeBetaServerEventResponseMCPCallArgumentsDelta {
  /**
   * The JSON-encoded arguments delta.
   */
  delta: string;
  /**
   * The unique ID of the server event.
   */
  event_id: string;
  /**
   * The ID of the MCP tool call item.
   */
  item_id: string;
  obfuscation?: string | unknown;
  /**
   * The index of the output item in the response.
   */
  output_index: number;
  /**
   * The ID of the response.
   */
  response_id: string;
  /**
   * The event type, must be `response.mcp_call_arguments.delta`.
   */
  type: string;
}