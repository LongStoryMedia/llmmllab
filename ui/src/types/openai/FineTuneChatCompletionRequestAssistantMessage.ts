

export interface FineTuneChatCompletionRequestAssistantMessage {
  /**
   * Controls whether the assistant message is trained against (0 or 1)
   */
  weight?: number;
  audio?: AudioOption0 | unknown;
  content?: string | (ChatCompletionRequestAssistantMessageContentPart)[] | unknown;
  function_call?: FunctionCallOption0 | unknown;
  /**
   * An optional name for the participant. Provides the model information to differentiate between participants of the same role.
   */
  name?: string;
  refusal?: string | unknown;
  /**
   * The role of the messages author, in this case `assistant`.
   */
  role: 'assistant';
  tool_calls?: ChatCompletionMessageToolCalls;
}