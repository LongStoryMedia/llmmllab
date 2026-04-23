

export interface CreateVoiceConsentRequest {
  /**
   * The BCP 47 language tag for the consent phrase (for example, `en-US`).
   */
  language: string;
  /**
   * The label to use for this consent recording.
   */
  name: string;
  /**
   * The consent audio recording file. Maximum size is 10 MiB.

Supported MIME types:
`audio/mpeg`, `audio/wav`, `audio/x-wav`, `audio/ogg`, `audio/aac`, `audio/flac`, `audio/webm`, `audio/mp4`.

   */
  recording: string;
}