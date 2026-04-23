

/**
 * Defines a function in your own code the model can choose to call. Learn more about [function calling](https://platform.openai.com/docs/guides/function-calling).
 */
export interface FunctionTool {
  description?: string | unknown;
  /**
   * The name of the function to call.
   */
  name: string;
  parameters: Record<string, unknown> | unknown;
  strict: boolean | unknown;
  /**
   * The type of the function tool. Always `function`.
   */
  type: 'function';
}