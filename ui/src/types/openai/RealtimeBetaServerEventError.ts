

/**
 * Details of the error.
 */
export interface Error {
  code?: string | unknown;
  event_id?: string | unknown;
  /**
   * A human-readable error message.
   */
  message: string;
  param?: string | unknown;
  /**
   * The type of error (e.g., "invalid_request_error", "server_error").

   */
  type: string;
}

/**
 * Returned when an error occurs, which could be a client problem or a server
problem. Most errors are recoverable and the session will stay open, we
recommend to implementors to monitor and log error messages by default.

 */
export interface RealtimeBetaServerEventError {
  /**
   * Details of the error.
   */
  error: Error;
  /**
   * The unique ID of the server event.
   */
  event_id: string;
  /**
   * The event type, must be `error`.
   */
  type: string;
}