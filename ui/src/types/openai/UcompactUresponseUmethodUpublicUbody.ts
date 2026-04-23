

import { ModelIdsCompaction } from './ModelIdsCompaction';

import { InputItem } from './InputItem';



export interface CompactResponseMethodPublicBody {
  input?: string | (InputItem)[] | unknown;
  instructions?: string | unknown;
  model: ModelIdsCompaction;
  previous_response_id?: string | unknown;
}