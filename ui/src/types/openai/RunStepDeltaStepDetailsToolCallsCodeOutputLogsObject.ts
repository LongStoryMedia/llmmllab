

/**
 * Text output from the Code Interpreter tool call as part of a run step.
 */
export interface RunStepDeltaStepDetailsToolCallsCodeOutputLogsObject {
  /**
   * The index of the output in the outputs array.
   */
  index: number;
  /**
   * The text output from the Code Interpreter tool call.
   */
  logs?: string;
  /**
   * Always `logs`.
   */
  type: 'logs';
}