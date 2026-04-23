

/**
 * Send this event to cancel an in-progress response. The server will respond 
with a `response.done` event with a status of `response.status=cancelled`. If 
there is no response to cancel, the server will respond with an error. It's safe
to call `response.cancel` even if no response is in progress, an error will be
returned the session will remain unaffected.

 */
export interface RealtimeClientEventResponseCancel {
  /**
   * Optional client-generated ID used to identify this event.
   */
  event_id?: string;
  /**
   * A specific response ID to cancel - if not provided, will cancel an 
in-progress response in the default conversation.

   */
  response_id?: string;
  /**
   * The event type, must be `response.cancel`.
   */
  type: string;
}