

import { VectorStoreSearchResultContentObject } from './VectorStoreSearchResultContentObject';

import { VectorStoreFileAttributes } from './VectorStoreFileAttributes';



export interface VectorStoreSearchResultItem {
  attributes: VectorStoreFileAttributes;
  /**
   * Content chunks from the file.
   */
  content: (VectorStoreSearchResultContentObject)[];
  /**
   * The ID of the vector store file.
   */
  file_id: string;
  /**
   * The name of the vector store file.
   */
  filename: string;
  /**
   * The similarity score for the result.
   */
  score: number;
}