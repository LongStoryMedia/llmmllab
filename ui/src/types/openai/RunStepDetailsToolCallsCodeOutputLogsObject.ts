

/**
 * Text output from the Code Interpreter tool call as part of a run step.
 */
export interface RunStepDetailsToolCallsCodeOutputLogsObject {
  /**
   * The text output from the Code Interpreter tool call.
   */
  logs: string;
  /**
   * Always `logs`.
   */
  type: 'logs';
}