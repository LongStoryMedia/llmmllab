

import { FunctionObject } from './FunctionObject';



export interface AssistantToolsFunction {
  function: FunctionObject;
  /**
   * The type of tool being defined: `function`
   */
  type: 'function';
}