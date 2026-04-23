

/**
 * A request for human approval of a tool invocation.

 */
export interface MCPApprovalRequest {
  /**
   * A JSON string of arguments for the tool.

   */
  arguments: string;
  /**
   * The unique ID of the approval request.

   */
  id: string;
  /**
   * The name of the tool to run.

   */
  name: string;
  /**
   * The label of the MCP server making the request.

   */
  server_label: string;
  /**
   * The type of the item. Always `mcp_approval_request`.

   */
  type: 'mcp_approval_request';
}