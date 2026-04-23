

import { ContainerMemoryLimit } from './ContainerMemoryLimit';



/**
 * Configuration for a code interpreter container. Optionally specify the IDs of the files to run the code on.
 */
export interface CodeInterpreterContainerAuto {
  /**
   * An optional list of uploaded files to make available to your code.
   */
  file_ids?: (string)[];
  memory_limit?: ContainerMemoryLimit | unknown;
  /**
   * Always `auto`.
   */
  type: 'auto';
}