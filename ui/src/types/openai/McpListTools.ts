

import { MCPListToolsTool } from './McpListToolsTool';



/**
 * A list of tools available on an MCP server.

 */
export interface MCPListTools {
  error?: string | unknown;
  /**
   * The unique ID of the list.

   */
  id: string;
  /**
   * The label of the MCP server.

   */
  server_label: string;
  /**
   * The tools available on the server.

   */
  tools: (MCPListToolsTool)[];
  /**
   * The type of the item. Always `mcp_list_tools`.

   */
  type: 'mcp_list_tools';
}