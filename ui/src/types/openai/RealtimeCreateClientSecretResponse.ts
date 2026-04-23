

import { RealtimeSessionCreateResponseGA } from './RealtimeSessionCreateResponseGa';

import { RealtimeTranscriptionSessionCreateResponseGA } from './RealtimeTranscriptionSessionCreateResponseGa';



/**
 * Response from creating a session and client secret for the Realtime API.

 */
export interface RealtimeCreateClientSecretResponse {
  /**
   * Expiration timestamp for the client secret, in seconds since epoch.
   */
  expires_at: number;
  /**
   * The session configuration for either a realtime or transcription session.

   */
  session: RealtimeSessionCreateResponseGA | RealtimeTranscriptionSessionCreateResponseGA;
  /**
   * The generated client secret value.
   */
  value: string;
}