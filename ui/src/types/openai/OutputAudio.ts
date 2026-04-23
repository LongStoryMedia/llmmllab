

/**
 * An audio output from the model.

 */
export interface OutputAudio {
  /**
   * Base64-encoded audio data from the model.

   */
  data: string;
  /**
   * The transcript of the audio data from the model.

   */
  transcript: string;
  /**
   * The type of the output audio. Always `output_audio`.

   */
  type: 'output_audio';
}