

import { Metadata } from './Metadata';



/**
 * A StoredCompletionsRunDataSource configuration describing a set of filters

 */
export interface EvalStoredCompletionsSource {
  created_after?: number | unknown;
  created_before?: number | unknown;
  limit?: number | unknown;
  metadata?: Metadata;
  model?: string | unknown;
  /**
   * The type of source. Always `stored_completions`.
   */
  type: 'stored_completions';
}