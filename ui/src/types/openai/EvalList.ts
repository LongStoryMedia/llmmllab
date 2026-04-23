

import { Eval } from './Eval';



/**
 * An object representing a list of evals.

 */
export interface EvalList {
  /**
   * An array of eval objects.

   */
  data: (Eval)[];
  /**
   * The identifier of the first eval in the data array.
   */
  first_id: string;
  /**
   * Indicates whether there are more evals available.
   */
  has_more: boolean;
  /**
   * The identifier of the last eval in the data array.
   */
  last_id: string;
  /**
   * The type of this object. It is always set to "list".

   */
  object: 'list';
}