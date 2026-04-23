

/**
 * The per-line object of the batch input file
 */
export interface BatchRequestInput {
  /**
   * A developer-provided per-request id that will be used to match outputs to inputs. Must be unique for each request in a batch.
   */
  custom_id?: string;
  /**
   * The HTTP method to be used for the request. Currently only `POST` is supported.
   */
  method?: 'POST';
  /**
   * The OpenAI API relative URL to be used for the request. Currently `/v1/responses`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/completions`, and `/v1/moderations` are supported.
   */
  url?: string;
}