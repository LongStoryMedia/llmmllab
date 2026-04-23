

import { RealtimeTranscriptionSessionCreateResponse } from './RealtimeTranscriptionSessionCreateResponse';



/**
 * Returned when a transcription session is created.

 */
export interface RealtimeBetaServerEventTranscriptionSessionCreated {
  /**
   * The unique ID of the server event.
   */
  event_id: string;
  session: RealtimeTranscriptionSessionCreateResponse;
  /**
   * The event type, must be `transcription_session.created`.
   */
  type: string;
}