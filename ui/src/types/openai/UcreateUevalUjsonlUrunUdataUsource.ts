

import { EvalJsonlFileContentSource } from './EvalJsonlFileContentSource';

import { EvalJsonlFileIdSource } from './EvalJsonlFileIdSource';



/**
 * A JsonlRunDataSource object with that specifies a JSONL file that matches the eval 

 */
export interface CreateEvalJsonlRunDataSource {
  /**
   * Determines what populates the `item` namespace in the data source.
   */
  source: EvalJsonlFileContentSource | EvalJsonlFileIdSource;
  /**
   * The type of data source. Always `jsonl`.
   */
  type: 'jsonl';
}