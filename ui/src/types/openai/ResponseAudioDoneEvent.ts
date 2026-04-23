

/**
 * Emitted when the audio response is complete.
 */
export interface ResponseAudioDoneEvent {
  /**
   * The sequence number of the delta.

   */
  sequence_number: number;
  /**
   * The type of the event. Always `response.audio.done`.

   */
  type: 'response.audio.done';
}