

import { VectorStoreExpirationAfter } from './VectorStoreExpirationAfter';

import { Metadata } from './Metadata';



export interface FileCounts {
  /**
   * The number of files that were cancelled.
   */
  cancelled: number;
  /**
   * The number of files that have been successfully processed.
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
 * A vector store is a collection of processed files can be used by the `file_search` tool.
 */
export interface VectorStoreObject {
  /**
   * The Unix timestamp (in seconds) for when the vector store was created.
   */
  created_at: number;
  expires_after?: VectorStoreExpirationAfter;
  expires_at?: number | unknown;
  file_counts: FileCounts;
  /**
   * The identifier, which can be referenced in API endpoints.
   */
  id: string;
  last_active_at: number | unknown;
  metadata: Metadata;
  /**
   * The name of the vector store.
   */
  name: string;
  /**
   * The object type, which is always `vector_store`.
   */
  object: 'vector_store';
  /**
   * The status of the vector store, which can be either `expired`, `in_progress`, or `completed`. A status of `completed` indicates that the vector store is ready for use.
   */
  status: 'expired' | 'in_progress' | 'completed';
  /**
   * The total number of bytes used by the files in the vector store.
   */
  usage_bytes: number;
}