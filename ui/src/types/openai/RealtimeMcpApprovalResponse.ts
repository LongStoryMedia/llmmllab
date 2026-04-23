

/**
 * A Realtime item responding to an MCP approval request.

 */
export interface RealtimeMCPApprovalResponse {
  /**
   * The ID of the approval request being answered.
   */
  approval_request_id: string;
  /**
   * Whether the request was approved.
   */
  approve: boolean;
  /**
   * The unique ID of the approval response.
   */
  id: string;
  reason?: string | unknown;
  /**
   * The type of the item. Always `mcp_approval_response`.
   */
  type: 'mcp_approval_response';
}