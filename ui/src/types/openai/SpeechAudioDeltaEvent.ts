

/**
 * Emitted for each chunk of audio data generated during speech synthesis.
 */
export interface SpeechAudioDeltaEvent {
  /**
   * A chunk of Base64-encoded audio data.

   */
  audio: string;
  /**
   * The type of the event. Always `speech.audio.delta`.

   */
  type: 'speech.audio.delta';
}