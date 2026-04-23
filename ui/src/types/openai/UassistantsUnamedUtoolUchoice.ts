

export interface Function {
  /**
   * The name of the function to call.
   */
  name: string;
}

/**
 * Specifies a tool the model should use. Use to force the model to call a specific tool.
 */
export interface AssistantsNamedToolChoice {
  function?: Function;
  /**
   * The type of the tool. If type is `function`, the function name must be set
   */
  type: 'function' | 'code_interpreter' | 'file_search';
}