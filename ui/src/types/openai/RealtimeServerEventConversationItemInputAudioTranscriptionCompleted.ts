

import { LogProbProperties } from './LogProbProperties';

import { TranscriptTextUsageTokens } from './TranscriptTextUsageTokens';

import { TranscriptTextUsageDuration } from './TranscriptTextUsageDuration';



/**
 * This event is the output of audio transcription for user audio written to the
user audio buffer. Transcription begins when the input audio buffer is
committed by the client or server (when VAD is enabled). Transcription runs
asynchronously with Response creation, so this event may come before or after
the Response events.

Realtime API models accept audio natively, and thus input transcription is a
separate process run on a separate ASR (Automatic Speech Recognition) model.
The transcript may diverge somewhat from the model's interpretation, and
should be treated as a rough guide.

 */
export interface RealtimeServerEventConversationItemInputAudioTranscriptionCompleted {
  /**
   * The index of the content part containing the audio.
   */
  content_index: number;
  /**
   * The unique ID of the server event.
   */
  event_id: string;
  /**
   * The ID of the item containing the audio that is being transcribed.
   */
  item_id: string;
  logprobs?: (LogProbProperties)[] | unknown;
  /**
   * The transcribed text.
   */
  transcript: string;
  /**
   * The event type, must be
`conversation.item.input_audio_transcription.completed`.

   */
  type: 'conversation.item.input_audio_transcription.completed';
  /**
   * Usage statistics for the transcription, this is billed according to the ASR model's pricing rather than the realtime model's pricing.
   */
  usage: TranscriptTextUsageTokens | TranscriptTextUsageDuration;
}