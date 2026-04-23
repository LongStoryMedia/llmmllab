

/**
 * Indicates that the shell call exceeded its configured time limit.
 */
export interface FunctionShellCallOutputTimeoutOutcome {
  /**
   * The outcome type. Always `timeout`.
   */
  type: 'timeout';
}