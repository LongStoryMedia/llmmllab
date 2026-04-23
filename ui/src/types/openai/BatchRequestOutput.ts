

/**
 * The per-line object of the batch output and error files
 */
export interface BatchRequestOutput {
  /**
   * A developer-provided per-request id that will be used to match outputs to inputs.
   */
  custom_id?: string;
  error?: ErrorOption0 | unknown;
  id?: string;
  response?: ResponseOption0 | unknown;
}