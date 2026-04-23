

import { Metadata } from './Metadata';

import { VectorStoreExpirationAfter } from './VectorStoreExpirationAfter';



export interface UpdateVectorStoreRequest {
  expires_after?: VectorStoreExpirationAfter & string;
  metadata?: Metadata;
  /**
   * The name of the vector store.
   */
  name?: string;
}