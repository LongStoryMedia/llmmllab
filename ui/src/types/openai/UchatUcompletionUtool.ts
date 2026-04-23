

import { FunctionObject } from './FunctionObject';



/**
 * A function tool that can be used to generate a response.

 */
export interface ChatCompletionTool {
  function: FunctionObject;
  /**
   * The type of the tool. Currently, only `function` is supported.
   */
  type: 'function';
}