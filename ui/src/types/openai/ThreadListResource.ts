

import { ThreadResource } from './ThreadResource';



/**
 * A paginated list of ChatKit threads.
 */
export interface ThreadListResource {
  /**
   * A list of items
   */
  data: (ThreadResource)[];
  first_id: string | unknown;
  /**
   * Whether there are more items available.
   */
  has_more: boolean;
  last_id: string | unknown;
  /**
   * The type of object returned, must be `list`.
   */
  object: string;
}