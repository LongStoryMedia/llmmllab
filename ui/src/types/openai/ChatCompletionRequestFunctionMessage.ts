

export interface ChatCompletionRequestFunctionMessage {
  content: string | unknown;
  /**
   * The name of the function to call.
   */
  name: string;
  /**
   * The role of the messages author, in this case `function`.
   */
  role: 'function';
}