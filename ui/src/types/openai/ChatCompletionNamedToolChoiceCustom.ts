

export interface Custom {
  /**
   * The name of the custom tool to call.
   */
  name: string;
}

/**
 * Specifies a tool the model should use. Use to force the model to call a specific custom tool.
 */
export interface ChatCompletionNamedToolChoiceCustom {
  custom: Custom;
  /**
   * For custom tool calling, the type is always `custom`.
   */
  type: 'custom';
}