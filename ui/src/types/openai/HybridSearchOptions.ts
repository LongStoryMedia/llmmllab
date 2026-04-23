

export interface HybridSearchOptions {
  /**
   * The weight of the embedding in the reciprocal ranking fusion.
   */
  embedding_weight: number;
  /**
   * The weight of the text in the reciprocal ranking fusion.
   */
  text_weight: number;
}