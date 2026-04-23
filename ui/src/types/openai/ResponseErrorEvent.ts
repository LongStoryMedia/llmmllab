

/**
 * Emitted when an error occurs.
 */
export interface ResponseErrorEvent {
  code: string | unknown;
  /**
   * The error message.

   */
  message: string;
  param: string | unknown;
  /**
   * The sequence number of this event.
   */
  sequence_number: number;
  /**
   * The type of the event. Always `error`.

   */
  type: 'error';
}