

/**
 * The output of a code interpreter tool call that is a file.

 */
export interface CodeInterpreterFileOutput {
  files: (FilesItem)[];
  /**
   * The type of the code interpreter file output. Always `files`.

   */
  type: 'files';
}