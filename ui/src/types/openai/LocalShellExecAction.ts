

/**
 * Execute a shell command on the server.
 */
export interface LocalShellExecAction {
  /**
   * The command to run.
   */
  command: (string)[];
  /**
   * Environment variables to set for the command.
   */
  env: Record<string, string>;
  timeout_ms?: number | unknown;
  /**
   * The type of the local shell action. Always `exec`.
   */
  type: 'exec';
  user?: string | unknown;
  working_directory?: string | unknown;
}