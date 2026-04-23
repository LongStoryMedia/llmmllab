

export interface Function {
  /**
   * The name of the function to call.
   */
  name: string;
}

/**
 * Specifies a tool the model should use. Use to force the model to call a specific function.
 */
export interface ChatCompletionNamedToolChoice {
  function: Function;
  /**
   * For function calling, the type is always `function`.
   */
  type: 'function';
}