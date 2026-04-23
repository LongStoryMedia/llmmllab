

/**
 * A pending safety check for the computer call.
 */
export interface ComputerCallSafetyCheckParam {
  code?: string | unknown;
  /**
   * The ID of the pending safety check.
   */
  id: string;
  message?: string | unknown;
}