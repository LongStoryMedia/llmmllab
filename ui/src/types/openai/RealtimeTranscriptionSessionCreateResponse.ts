

import { AudioTranscription } from './AudioTranscription';



/**
 * Ephemeral key returned by the API. Only present when the session is
created on the server via REST API.

 */
export interface ClientSecret {
  /**
   * Timestamp for when the token expires. Currently, all tokens expire
after one minute.

   */
  expires_at: number;
  /**
   * Ephemeral key usable in client environments to authenticate connections
to the Realtime API. Use this in client-side environments rather than
a standard API token, which should only be used server-side.

   */
  value: string;
}

/**
 * Configuration for turn detection. Can be set to `null` to turn off. Server
VAD means that the model will detect the start and end of speech based on
audio volume and respond at the end of user speech.

 */
export interface TurnDetection {
  /**
   * Amount of audio to include before the VAD detected speech (in
milliseconds). Defaults to 300ms.

   */
  prefix_padding_ms?: number;
  /**
   * Duration of silence to detect speech stop (in milliseconds). Defaults
to 500ms. With shorter values the model will respond more quickly,
but may jump in on short pauses from the user.

   */
  silence_duration_ms?: number;
  /**
   * Activation threshold for VAD (0.0 to 1.0), this defaults to 0.5. A
higher threshold will require louder audio to activate the model, and
thus might perform better in noisy environments.

   */
  threshold?: number;
  /**
   * Type of turn detection, only `server_vad` is currently supported.

   */
  type?: string;
}

/**
 * A new Realtime transcription session configuration.

When a session is created on the server via REST API, the session object
also contains an ephemeral key. Default TTL for keys is 10 minutes. This
property is not present when a session is updated via the WebSocket API.

 */
export interface RealtimeTranscriptionSessionCreateResponse {
  /**
   * Ephemeral key returned by the API. Only present when the session is
created on the server via REST API.

   */
  client_secret: ClientSecret;
  /**
   * The format of input audio. Options are `pcm16`, `g711_ulaw`, or `g711_alaw`.

   */
  input_audio_format?: string;
  /**
   * Configuration of the transcription model.

   */
  input_audio_transcription?: AudioTranscription;
  /**
   * The set of modalities the model can respond with. To disable audio,
set this to ["text"].

   */
  modalities?: string;
  /**
   * Configuration for turn detection. Can be set to `null` to turn off. Server
VAD means that the model will detect the start and end of speech based on
audio volume and respond at the end of user speech.

   */
  turn_detection?: TurnDetection;
}