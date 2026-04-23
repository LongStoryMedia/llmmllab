

import { TranscriptTextUsageTokens } from './TranscriptTextUsageTokens';



/**
 * Emitted when the transcription is complete. Contains the complete transcription text. Only emitted when you [create a transcription](https://platform.openai.com/docs/api-reference/audio/create-transcription) with the `Stream` parameter set to `true`.
 */
export interface TranscriptTextDoneEvent {
  /**
   * The log probabilities of the individual tokens in the transcription. Only included if you [create a transcription](https://platform.openai.com/docs/api-reference/audio/create-transcription) with the `include[]` parameter set to `logprobs`.

   */
  logprobs?: (LogprobsItem)[];
  /**
   * The text that was transcribed.

   */
  text: string;
  /**
   * The type of the event. Always `transcript.text.done`.

   */
  type: 'transcript.text.done';
  usage?: TranscriptTextUsageTokens;
}