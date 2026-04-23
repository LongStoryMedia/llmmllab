

import { ModerationImageURLInput } from './ModerationImageUrlInput';

import { ModerationTextInput } from './ModerationTextInput';



export interface CreateModerationRequest {
  /**
   * Input (or inputs) to classify. Can be a single string, an array of strings, or
an array of multi-modal input objects similar to other models.

   */
  input: string | (string)[] | (ModerationImageURLInput | ModerationTextInput)[];
  /**
   * The content moderation model you would like to use. Learn more in
[the moderation guide](https://platform.openai.com/docs/guides/moderation), and learn about
available models [here](https://platform.openai.com/docs/models#moderation).

   */
  model?: string | 'omni-moderation-latest' | 'omni-moderation-2024-09-26' | 'text-moderation-latest' | 'text-moderation-stable';
}