

/**
 * Use this option to force the model to call a specific custom tool.

 */
export interface ToolChoiceCustom {
  /**
   * The name of the custom tool to call.
   */
  name: string;
  /**
   * For custom tool calling, the type is always `custom`.
   */
  type: 'custom';
}