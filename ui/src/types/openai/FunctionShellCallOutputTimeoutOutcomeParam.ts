

/**
 * Indicates that the shell call exceeded its configured time limit.
 */
export interface FunctionShellCallOutputTimeoutOutcomeParam {
  /**
   * The outcome type. Always `timeout`.
   */
  type: 'timeout';
}