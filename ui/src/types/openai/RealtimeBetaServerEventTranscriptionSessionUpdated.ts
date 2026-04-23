

import { RealtimeTranscriptionSessionCreateResponse } from './RealtimeTranscriptionSessionCreateResponse';



/**
 * Returned when a transcription session is updated with a `transcription_session.update` event, unless 
there is an error.

 */
export interface RealtimeBetaServerEventTranscriptionSessionUpdated {
  /**
   * The unique ID of the server event.
   */
  event_id: string;
  session: RealtimeTranscriptionSessionCreateResponse;
  /**
   * The event type, must be `transcription_session.updated`.
   */
  type: string;
}