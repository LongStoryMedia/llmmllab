

/**
 * Execute a shell command.
 */
export interface FunctionShellAction {
  commands: (string)[];
  max_output_length: number | unknown;
  timeout_ms: number | unknown;
}