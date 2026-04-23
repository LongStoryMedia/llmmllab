

import { RealtimeSessionCreateRequestGA } from './RealtimeSessionCreateRequestGa';

import { RealtimeTranscriptionSessionCreateRequestGA } from './RealtimeTranscriptionSessionCreateRequestGa';



/**
 * Returned when a session is updated with a `session.update` event, unless
there is an error.

 */
export interface RealtimeServerEventSessionUpdated {
  /**
   * The unique ID of the server event.
   */
  event_id: string;
  /**
   * The session configuration.
   */
  session: RealtimeSessionCreateRequestGA | RealtimeTranscriptionSessionCreateRequestGA;
  /**
   * The event type, must be `session.updated`.
   */
  type: string;
}