

/**
 * Use this option to force the model to call a specific function.

 */
export interface ToolChoiceFunction {
  /**
   * The name of the function to call.
   */
  name: string;
  /**
   * For function calling, the type is always `function`.
   */
  type: 'function';
}