

export interface FileCounts {
  /**
   * The number of files that where cancelled.
   */
  cancelled: number;
  /**
   * The number of files that have been processed.
   */
  completed: number;
  /**
   * The number of files that have failed to process.
   */
  failed: number;
  /**
   * The number of files that are currently being processed.
   */
  in_progress: number;
  /**
   * The total number of files.
   */
  total: number;
}

/**
 * A batch of files attached to a vector store.
 */
export interface VectorStoreFileBatchObject {
  /**
   * The Unix timestamp (in seconds) for when the vector store files batch was created.
   */
  created_at: number;
  file_counts: FileCounts;
  /**
   * The identifier, which can be referenced in API endpoints.
   */
  id: string;
  /**
   * The object type, which is always `vector_store.file_batch`.
   */
  object: 'vector_store.files_batch';
  /**
   * The status of the vector store files batch, which can be either `in_progress`, `completed`, `cancelled` or `failed`.
   */
  status: 'in_progress' | 'completed' | 'cancelled' | 'failed';
  /**
   * The ID of the [vector store](https://platform.openai.com/docs/api-reference/vector-stores/object) that the [File](https://platform.openai.com/docs/api-reference/files) is attached to.
   */
  vector_store_id: string;
}