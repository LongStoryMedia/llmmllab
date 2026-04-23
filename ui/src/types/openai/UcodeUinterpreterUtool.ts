

import { CodeInterpreterContainerAuto } from './CodeInterpreterContainerAuto';



/**
 * A tool that runs Python code to help generate a response to a prompt.

 */
export interface CodeInterpreterTool {
  /**
   * The code interpreter container. Can be a container ID or an object that
specifies uploaded file IDs to make available to your code, along with an
optional `memory_limit` setting.

   */
  container: string | CodeInterpreterContainerAuto;
  /**
   * The type of the code interpreter tool. Always `code_interpreter`.

   */
  type: 'code_interpreter';
}