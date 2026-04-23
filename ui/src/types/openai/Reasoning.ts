

import { ReasoningEffort } from './ReasoningEffort';



/**
 * **gpt-5 and o-series models only**

Configuration options for
[reasoning models](https://platform.openai.com/docs/guides/reasoning).

 */
export interface Reasoning {
  effort?: ReasoningEffort;
  generate_summary?: 'auto' | 'concise' | 'detailed' | unknown;
  summary?: 'auto' | 'concise' | 'detailed' | unknown;
}