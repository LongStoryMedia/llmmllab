

/**
 * A tool that allows the model to execute shell commands in a local environment.
 */
export interface LocalShellToolParam {
  /**
   * The type of the local shell tool. Always `local_shell`.
   */
  type: 'local_shell';
}