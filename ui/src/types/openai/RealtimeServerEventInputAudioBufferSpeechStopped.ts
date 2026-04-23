

/**
 * Returned in `server_vad` mode when the server detects the end of speech in 
the audio buffer. The server will also send an `conversation.item.created` 
event with the user message item that is created from the audio buffer.

 */
export interface RealtimeServerEventInputAudioBufferSpeechStopped {
  /**
   * Milliseconds since the session started when speech stopped. This will 
correspond to the end of audio sent to the model, and thus includes the 
`min_silence_duration_ms` configured in the Session.

   */
  audio_end_ms: number;
  /**
   * The unique ID of the server event.
   */
  event_id: string;
  /**
   * The ID of the user message item that will be created.
   */
  item_id: string;
  /**
   * The event type, must be `input_audio_buffer.speech_stopped`.
   */
  type: string;
}