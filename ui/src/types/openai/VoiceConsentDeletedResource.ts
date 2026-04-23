

export interface VoiceConsentDeletedResource {
  deleted: boolean;
  /**
   * The consent recording identifier.
   */
  id: string;
  object: 'audio.voice_consent';
}