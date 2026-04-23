

import { ComparisonFilter } from './ComparisonFilter';



/**
 * Combine multiple filters using `and` or `or`.
 */
export interface CompoundFilter {
  /**
   * Array of filters to combine. Items can be `ComparisonFilter` or `CompoundFilter`.
   */
  filters: (ComparisonFilter | string)[];
  /**
   * Type of operation: `and` or `or`.
   */
  type: 'and' | 'or';
}