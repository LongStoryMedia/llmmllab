

/**
 * Details of the transcription error.
 */
export interface Error {
  /**
   * Error code, if any.
   */
  code?: string;
  /**
   * A human-readable error message.
   */
  message?: string;
  /**
   * Parameter related to the error, if any.
   */
  param?: string;
  /**
   * The type of error.
   */
  type?: string;
}

/**
 * Returned when input audio transcription is configured, and a transcription 
request for a user message failed. These events are separate from other 
`error` events so that the client can identify the related Item.

 */
export interface RealtimeServerEventConversationItemInputAudioTranscriptionFailed {
  /**
   * The index of the content part containing the audio.
   */
  content_index: number;
  /**
   * Details of the transcription error.
   */
  error: Error;
  /**
   * The unique ID of the server event.
   */
  event_id: string;
  /**
   * The ID of the user message item.
   */
  item_id: string;
  /**
   * The event type, must be
`conversation.item.input_audio_transcription.failed`.

   */
  type: 'conversation.item.input_audio_transcription.failed';
}