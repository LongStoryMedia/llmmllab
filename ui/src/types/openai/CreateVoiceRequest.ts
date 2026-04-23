

export interface CreateVoiceRequest {
  /**
   * The sample audio recording file. Maximum size is 10 MiB.

Supported MIME types:
`audio/mpeg`, `audio/wav`, `audio/x-wav`, `audio/ogg`, `audio/aac`, `audio/flac`, `audio/webm`, `audio/mp4`.

   */
  audio_sample: string;
  /**
   * The consent recording ID (for example, `cons_1234`).
   */
  consent: string;
  /**
   * The name of the new voice.
   */
  name: string;
}