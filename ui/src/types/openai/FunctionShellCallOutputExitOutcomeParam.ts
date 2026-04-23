

/**
 * Indicates that the shell commands finished and returned an exit code.
 */
export interface FunctionShellCallOutputExitOutcomeParam {
  /**
   * The exit code returned by the shell process.
   */
  exit_code: number;
  /**
   * The outcome type. Always `exit`.
   */
  type: 'exit';
}