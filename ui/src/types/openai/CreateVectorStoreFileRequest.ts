

import { VectorStoreFileAttributes } from './VectorStoreFileAttributes';

import { ChunkingStrategyRequestParam } from './ChunkingStrategyRequestParam';



export interface CreateVectorStoreFileRequest {
  attributes?: VectorStoreFileAttributes;
  chunking_strategy?: ChunkingStrategyRequestParam;
  /**
   * A [File](https://platform.openai.com/docs/api-reference/files) ID that the vector store should use. Useful for tools like `file_search` that can access files.
   */
  file_id: string;
}