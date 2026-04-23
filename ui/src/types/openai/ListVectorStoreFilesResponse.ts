

import { VectorStoreFileObject } from './VectorStoreFileObject';



export interface ListVectorStoreFilesResponse {
  data: (VectorStoreFileObject)[];
  first_id: string;
  has_more: boolean;
  last_id: string;
  object: string;
}