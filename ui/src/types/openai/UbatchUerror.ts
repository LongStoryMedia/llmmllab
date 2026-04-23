

export interface BatchError {
  /**
   * An error code identifying the error type.
   */
  code?: string;
  line?: number | unknown;
  /**
   * A human-readable message providing more details about the error.
   */
  message?: string;
  param?: string | unknown;
}