

/**
 * Deprecated in favor of LogsDataSourceConfig.

 */
export interface CreateEvalStoredCompletionsDataSourceConfig {
  /**
   * Metadata filters for the stored completions data source.
   */
  metadata?: Record<string, unknown>;
  /**
   * The type of data source. Always `stored_completions`.
   */
  type: 'stored_completions';
}