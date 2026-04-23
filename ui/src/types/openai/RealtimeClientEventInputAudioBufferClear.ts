

/**
 * Send this event to clear the audio bytes in the buffer. The server will 
respond with an `input_audio_buffer.cleared` event.

 */
export interface RealtimeClientEventInputAudioBufferClear {
  /**
   * Optional client-generated ID used to identify this event.
   */
  event_id?: string;
  /**
   * The event type, must be `input_audio_buffer.clear`.
   */
  type: string;
}