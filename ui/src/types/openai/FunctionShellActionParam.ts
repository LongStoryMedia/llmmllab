

/**
 * Commands and limits describing how to run the shell tool call.
 */
export interface FunctionShellActionParam {
  /**
   * Ordered shell commands for the execution environment to run.
   */
  commands: (string)[];
  max_output_length?: number | unknown;
  timeout_ms?: number | unknown;
}