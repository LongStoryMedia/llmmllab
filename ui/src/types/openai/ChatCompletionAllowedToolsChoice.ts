

import { ChatCompletionAllowedTools } from './ChatCompletionAllowedTools';



/**
 * Constrains the tools available to the model to a pre-defined set.

 */
export interface ChatCompletionAllowedToolsChoice {
  allowed_tools: ChatCompletionAllowedTools;
  /**
   * Allowed tool configuration type. Always `allowed_tools`.
   */
  type: 'allowed_tools';
}