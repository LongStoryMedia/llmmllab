

import { Metadata } from './Metadata';



/**
 * Deprecated in favor of LogsDataSourceConfig.

 */
export interface EvalStoredCompletionsDataSourceConfig {
  metadata?: Metadata;
  /**
   * The json schema for the run data source items.
Learn how to build JSON schemas [here](https://json-schema.org/).

   */
  schema: Record<string, unknown>;
  /**
   * The type of data source. Always `stored_completions`.
   */
  type: 'stored_completions';
}