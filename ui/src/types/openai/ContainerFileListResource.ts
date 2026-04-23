

import { ContainerFileResource } from './ContainerFileResource';



export interface ContainerFileListResource {
  /**
   * A list of container files.
   */
  data: (ContainerFileResource)[];
  /**
   * The ID of the first file in the list.
   */
  first_id: string;
  /**
   * Whether there are more files available.
   */
  has_more: boolean;
  /**
   * The ID of the last file in the list.
   */
  last_id: string;
  /**
   * The type of object returned, must be 'list'.
   */
  object: string;
}