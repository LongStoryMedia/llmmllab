

import { RealtimeSession } from './RealtimeSession';



/**
 * Returned when a session is updated with a `session.update` event, unless
there is an error.

 */
export interface RealtimeBetaServerEventSessionUpdated {
  /**
   * The unique ID of the server event.
   */
  event_id: string;
  session: RealtimeSession;
  /**
   * The event type, must be `session.updated`.
   */
  type: string;
}