

/**
 * The logs output from the code interpreter.
 */
export interface CodeInterpreterOutputLogs {
  /**
   * The logs output from the code interpreter.
   */
  logs: string;
  /**
   * The type of the output. Always `logs`.
   */
  type: 'logs';
}