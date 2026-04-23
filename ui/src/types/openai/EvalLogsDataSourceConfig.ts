

import { Metadata } from './Metadata';



/**
 * A LogsDataSourceConfig which specifies the metadata property of your logs query.
This is usually metadata like `usecase=chatbot` or `prompt-version=v2`, etc.
The schema returned by this data source config is used to defined what variables are available in your evals.
`item` and `sample` are both defined when using this data source config.

 */
export interface EvalLogsDataSourceConfig {
  metadata?: Metadata;
  /**
   * The json schema for the run data source items.
Learn how to build JSON schemas [here](https://json-schema.org/).

   */
  schema: Record<string, unknown>;
  /**
   * The type of data source. Always `logs`.
   */
  type: 'logs';
}