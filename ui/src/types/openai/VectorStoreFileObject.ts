

import { VectorStoreFileAttributes } from './VectorStoreFileAttributes';

import { ChunkingStrategyResponse } from './ChunkingStrategyResponse';



/**
 * A list of files attached to a vector store.
 */
export interface VectorStoreFileObject {
  attributes?: VectorStoreFileAttributes;
  chunking_strategy?: ChunkingStrategyResponse;
  /**
   * The Unix timestamp (in seconds) for when the vector store file was created.
   */
  created_at: number;
  /**
   * The identifier, which can be referenced in API endpoints.
   */
  id: string;
  last_error: LastErrorOption0 | unknown;
  /**
   * The object type, which is always `vector_store.file`.
   */
  object: 'vector_store.file';
  /**
   * The status of the vector store file, which can be either `in_progress`, `completed`, `cancelled`, or `failed`. The status `completed` indicates that the vector store file is ready for use.
   */
  status: 'in_progress' | 'completed' | 'cancelled' | 'failed';
  /**
   * The total vector store usage in bytes. Note that this may be different from the original file size.
   */
  usage_bytes: number;
  /**
   * The ID of the [vector store](https://platform.openai.com/docs/api-reference/vector-stores/object) that the [File](https://platform.openai.com/docs/api-reference/files) is attached to.
   */
  vector_store_id: string;
}