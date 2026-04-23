

/**
 * A data source config which specifies the metadata property of your logs query.
This is usually metadata like `usecase=chatbot` or `prompt-version=v2`, etc.

 */
export interface CreateEvalLogsDataSourceConfig {
  /**
   * Metadata filters for the logs data source.
   */
  metadata?: Record<string, unknown>;
  /**
   * The type of data source. Always `logs`.
   */
  type: 'logs';
}