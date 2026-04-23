

/**
 * This is returned when the chunking strategy is unknown. Typically, this is because the file was indexed before the `chunking_strategy` concept was introduced in the API.
 */
export interface OtherChunkingStrategyResponseParam {
  /**
   * Always `other`.
   */
  type: 'other';
}