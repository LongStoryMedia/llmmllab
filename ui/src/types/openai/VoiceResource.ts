

/**
 * A custom voice that can be used for audio output.
 */
export interface VoiceResource {
  /**
   * The Unix timestamp (in seconds) for when the voice was created.
   */
  created_at: number;
  /**
   * The voice identifier, which can be referenced in API endpoints.
   */
  id: string;
  /**
   * The name of the voice.
   */
  name: string;
  /**
   * The object type, which is always `audio.voice`.
   */
  object: 'audio.voice';
}