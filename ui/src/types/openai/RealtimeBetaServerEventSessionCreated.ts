

import { RealtimeSession } from './RealtimeSession';



/**
 * Returned when a Session is created. Emitted automatically when a new
connection is established as the first server event. This event will contain
the default Session configuration.

 */
export interface RealtimeBetaServerEventSessionCreated {
  /**
   * The unique ID of the server event.
   */
  event_id: string;
  session: RealtimeSession;
  /**
   * The event type, must be `session.created`.
   */
  type: string;
}