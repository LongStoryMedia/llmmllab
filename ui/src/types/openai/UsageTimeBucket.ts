

import { UsageCompletionsResult } from './UsageCompletionsResult';

import { UsageEmbeddingsResult } from './UsageEmbeddingsResult';

import { UsageModerationsResult } from './UsageModerationsResult';

import { UsageImagesResult } from './UsageImagesResult';

import { UsageAudioSpeechesResult } from './UsageAudioSpeechesResult';

import { UsageAudioTranscriptionsResult } from './UsageAudioTranscriptionsResult';

import { UsageVectorStoresResult } from './UsageVectorStoresResult';

import { UsageCodeInterpreterSessionsResult } from './UsageCodeInterpreterSessionsResult';

import { CostsResult } from './CostsResult';



export interface UsageTimeBucket {
  end_time: number;
  object: 'bucket';
  result: (UsageCompletionsResult | UsageEmbeddingsResult | UsageModerationsResult | UsageImagesResult | UsageAudioSpeechesResult | UsageAudioTranscriptionsResult | UsageVectorStoresResult | UsageCodeInterpreterSessionsResult | CostsResult)[];
  start_time: number;
}